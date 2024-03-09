import os
import logging
import random
import json
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, SwinModel
from PIL import Image, ImageFile
import pandas as pd
import sys
from argparse import ArgumentParser
np.set_printoptions(threshold=sys.maxsize)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def logger_config(log_path,logging_name):
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class PairDataset(Dataset):

    def __init__(self, data, image_path, item_imagename_dict, feature_extractor):
        self.data = data
        self.image_path = image_path
        self.item_imagename_dict = item_imagename_dict
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def getimage(self, src_id, tgt_id):
        src_image_name = self.item_imagename_dict[src_id]
        tgt_image_name = self.item_imagename_dict[tgt_id]
        src_image = Image.open(os.path.join(self.image_path, src_image_name)).convert("RGB")
        tgt_image = Image.open(os.path.join(self.image_path, tgt_image_name)).convert("RGB")
        return src_image, tgt_image

    def collate(self, examples):
        batch_src_images, batch_tgt_images = [], []
        batch_src_item_ids, batch_tgt_item_ids = [], []
        
        for item in examples:
            src_image, tgt_image = self.getimage(item["src_item_id"], item["tgt_item_id"])
            src_image = np.array(self.feature_extractor(src_image, return_tensors="pt").pixel_values.squeeze(0))
            tgt_image = np.array(self.feature_extractor(tgt_image, return_tensors="pt").pixel_values.squeeze(0))
            batch_src_images.append(src_image)
            batch_tgt_images.append(tgt_image)

            batch_src_item_ids.append(item['src_item_id'])
            batch_tgt_item_ids.append(item['tgt_item_id'])
        
        return batch_src_item_ids, batch_tgt_item_ids, torch.tensor(batch_src_images), torch.tensor(batch_tgt_images)

    def __getitem__(self, index):
        item = self.data[index]
        return item


class Baseline(nn.Module):

    def __init__(self, encoder, hidden_size, output_size):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, image):
        outputs = self.encoder(image)
        outputs = self.linear(outputs.pooler_output)
        # outputs = F.normalize(outputs)
        return outputs


class MetricsCalculator(object):

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def get_tp(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, self.beta).float()
        return torch.sum(y_pred[y_true == 1])

    def get_fp(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, self.beta).float()
        return torch.sum(y_pred[y_true == 0])

    def get_fn(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, self.beta).float()
        return torch.sum(y_true) - torch.sum(y_pred[y_true == 1])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--datafolder", type=str, default="data")
    parser.add_argument("--image_path", type=str, default="data/item_test_images")
    parser.add_argument("--save_path", type=str, default="output/swin_large384_warmupdecay")
    parser.add_argument("--save_csv_name", type=str, default="pair_fea_list_image_512.csv")
    parser.add_argument("--pretrain_image_model_path", type=str, default="../../../pretrained_model/swin_large_patch4_window12_384_in22k")
    parser.add_argument("--image_model_name", type=str, default="baseline_image_large_512_23.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.76)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")

    args = parser.parse_args()
    batch_size = args.batch_size
    threshold = args.threshold
    torch.set_num_threads(args.num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES
    device = torch.device('cuda:0')

    datafolder = args.datafolder
    image_path = args.image_path
    save_path = args.save_path

    with open(os.path.join(datafolder, 'item_test_info.json'), 'r') as file:
        item_valid_info = json.load(file)
    with open(os.path.join(datafolder, 'item_test_pair.json'), 'r') as file:
        item_valid_pair = json.load(file)

    item_imagename_dict = {}
    for item in item_valid_info:
        item_imagename_dict[item["item_id"]] = item["item_image_name"]

    image_model_path = args.pretrain_image_model_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_path)
    valid_pair = PairDataset(data=item_valid_pair, image_path=image_path, item_imagename_dict=item_imagename_dict, feature_extractor=feature_extractor)
    valid_load_pair = DataLoader(valid_pair, batch_size=batch_size, collate_fn=valid_pair.collate, num_workers=8)

    encoder = SwinModel.from_pretrained(image_model_path)
    model = Baseline(encoder=encoder, hidden_size=1536, output_size=512).to(device)
    model.load_state_dict(torch.load(os.path.join(save_path, args.image_model_name), map_location=torch.device('cuda:0')))

    model.eval()

    pair_fea_list = []
    with torch.no_grad():
        for batch in tqdm(valid_load_pair):
            batch_src_item_ids, batch_tgt_item_ids, batch_src_images, batch_tgt_images = batch

            batch_src_images, batch_tgt_images = batch_src_images.to(device), batch_tgt_images.to(device)
            
            src_fea = model(batch_src_images)
            tgt_fea = model(batch_tgt_images)

            for src_item_id, tgt_item_id, src, tgt in zip(batch_src_item_ids, batch_tgt_item_ids, src_fea, tgt_fea):
                pair_fea_list.append([src_item_id, src.detach().cpu().numpy(), tgt_item_id, tgt.detach().cpu().numpy(), threshold])
        result = pd.DataFrame(columns=["src_item_id", "src_item_emb", "tgt_item_id", "tgt_item_emb", "threshold"], data=pair_fea_list)
        result.to_csv(args.save_csv_name)
        print("predict and save success!")
