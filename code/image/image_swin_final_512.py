from torch.cuda.amp import autocast, GradScaler
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
from transformers import AutoFeatureExtractor, SwinModel, get_linear_schedule_with_warmup
from PIL import Image, ImageFile
from argparse import ArgumentParser
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


# class ItemDataset(Dataset):

#     def __init__(self, data, image_path, item_imagename_dict, feature_extractor, transform=None):
#         self.data = data
#         self.image_path = image_path
#         self.transform = transform
#         self.item_imagename_dict = item_imagename_dict
#         self.feature_extractor = feature_extractor

#     def __len__(self):
#         return len(self.data)

#     def getimage(self, item_id):
#         image_name = item_imagename_dict[item_id]
#         image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
#         return item_id, image

#     def collate(self, examples):
#         batch_item_ids, batch_item_images = [], []
#         for item in examples:
#             item_id, image = self.getimage(item["item_id"])
#             image = np.array(self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze(0))
#             batch_item_ids.append(item_id)
#             batch_item_images.append(image)
#         return batch_item_ids, batch_item_images

#     def __getitem__(self, index):
#         item = self.data[index]
#         return item


class PairDataset(Dataset):

    def __init__(self, data, image_path, item_imagename_dict, feature_extractor):
        self.data = data
        self.image_path = image_path
        self.item_imagename_dict = item_imagename_dict
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def getimage(self, item_id):
        image_name = self.item_imagename_dict[item_id]
        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        return image

    def collate(self, examples):
        batch_src_images, batch_tgt_images = [], []
        batch_label = []
        
        for item in examples:
            src_image = self.getimage(item["src_item_id"])
            tgt_image = self.getimage(item["tgt_item_id"])
            src_image = self.feature_extractor(src_image, return_tensors="np").pixel_values.squeeze(0)
            tgt_image = self.feature_extractor(tgt_image, return_tensors="np").pixel_values.squeeze(0)
            batch_src_images.append(src_image)
            batch_tgt_images.append(tgt_image)

            batch_label.append(int(item["item_label"]))
        
        return torch.tensor(batch_src_images), torch.tensor(batch_tgt_images), torch.tensor(batch_label).float()

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
        return outputs


class CoSENT(object):
    def __init__(self, device):
        self.device = device
        super().__init__()

    def loss_function(self, y_pred_src, y_pred_tgt, y_true):
        # 1. 取出真实的标签
        # y_true = y_true[::2]    # tensor([1, 0, 1]) 真实的标签

        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms_src = (y_pred_src ** 2).sum(axis=1, keepdims=True) ** 0.5
        norms_tgt = (y_pred_tgt ** 2).sum(axis=1, keepdims=True) ** 0.5
        # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
        y_pred_src = y_pred_src / norms_src
        y_pred_tgt = y_pred_tgt / norms_tgt

        # 3. 奇偶向量相乘
        y_pred = torch.sum(y_pred_src * y_pred_tgt, dim=1) * 20

        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        if torch.cuda.is_available():
            y_pred = torch.cat((torch.tensor([0]).float().to(self.device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        else:
            y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        
        return torch.logsumexp(y_pred, dim=0)


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


if __name__ == "__main__":
    logger = logger_config(log_path='log_swin_large384_warmupdecay_512.txt', logging_name='baseline_swin_large384_warmupdecay_512')
    parser = ArgumentParser()
    parser.add_argument("--datafolder", type=str, default="data")
    parser.add_argument("--image_path", type=str, default="data/item_train_images")
    parser.add_argument("--save_path", type=str, default="output/swin_large384_warmupdecay")
    parser.add_argument("--pretrain_image_model_path", type=str, default="../../../pretrained_model/swin_large_patch4_window12_384_in22k")
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.76)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--warm_up_ratio", type=float, default=0.1)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")

    args = parser.parse_args()
    epochs = args.num_train_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    threshold = args.threshold
    warm_up_ratio = args.warm_up_ratio
    torch.set_num_threads(args.num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES
    device = torch.device('cuda:0')

    datafolder = args.datafolder
    image_path = args.image_path
    save_path = args.save_path
    with open(os.path.join(datafolder, 'item_train_info.json'), 'r') as file:
        item_train_info = json.load(file)
    with open(os.path.join(datafolder, 'item_train_pair.json'), 'r') as file:
        item_train_pair = json.load(file)

    item_imagename_dict = {}
    for item in item_train_info:
        item_imagename_dict[item["item_id"]] = item["item_image_name"]

    random.seed(args.seed)
    random.shuffle(item_train_pair)
    train_pair = item_train_pair[:int(0.8 * len(item_train_pair))]
    valid_pair = item_train_pair[int(0.8 * len(item_train_pair)):int(0.9 * len(item_train_pair))]
    test_pair = item_train_pair[int(0.9 * len(item_train_pair)):]

    image_model_path = args.pretrain_image_model_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_path)

    train_pair = PairDataset(data=train_pair, image_path=image_path, item_imagename_dict=item_imagename_dict, feature_extractor=feature_extractor)
    train_load_pair = DataLoader(train_pair, batch_size=batch_size, collate_fn=train_pair.collate, num_workers=8, pin_memory=True)
    valid_pair = PairDataset(data=valid_pair, image_path=image_path, item_imagename_dict=item_imagename_dict, feature_extractor=feature_extractor)
    valid_load_pair = DataLoader(valid_pair, batch_size=batch_size, collate_fn=valid_pair.collate, num_workers=8, pin_memory=True)
    test_pair = PairDataset(data=test_pair, image_path=image_path, item_imagename_dict=item_imagename_dict, feature_extractor=feature_extractor)
    test_load_pair = DataLoader(test_pair, batch_size=batch_size, collate_fn=test_pair.collate, num_workers=8, pin_memory=True)
    
    encoder = SwinModel.from_pretrained(image_model_path)
    model = Baseline(encoder=encoder, hidden_size=1536, output_size=512).to(device)
    # model.load_state_dict(torch.load(os.path.join(save_path, 'baseline_image_swin2.pkl')))
    
    unfreeze_layers = ['layers.2.', 'layers.3.', 'encoder.layernorm.', 'linear.']
    # unfreeze_layers = ['layers.2.blocks.12.', 'layers.2.blocks.13.', 'layers.2.blocks.14.', 'layers.2.blocks.15.', 'layers.2.blocks.16.', 'layers.2.blocks.17.', 'layers.2.downsample.', 'layers.3.', 'encoder.layernorm.', 'linear.']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    total_steps = len(train_load_pair) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps, num_training_steps=total_steps)

    cosent = CoSENT(device=device)
    metrics = MetricsCalculator(beta=threshold)
    scaler = GradScaler()

    max_f1 = 0.0
    for epoch in range(epochs):
        # train
        total_loss, total_tp, total_fp, total_fn = 0.0, 0.0, 0.0, 0.0
        for batch in tqdm(train_load_pair):
            model.train()
            batch_src_images, batch_tgt_images, batch_label = batch
            batch_src_images, batch_tgt_images, batch_label = batch_src_images.to(device), batch_tgt_images.to(device), batch_label.to(device)
            
            with autocast():
                src_fea = model(batch_src_images)
                tgt_fea = model(batch_tgt_images)
                loss = cosent.loss_function(src_fea, tgt_fea, batch_label)
            total_loss += loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.eval()
            cos_sim = torch.cosine_similarity(src_fea, tgt_fea)
            cos_sim = (cos_sim + 1) * 0.5
            total_tp += metrics.get_tp(cos_sim, batch_label)
            total_fn += metrics.get_fn(cos_sim, batch_label)
            total_fp += metrics.get_fp(cos_sim, batch_label)
        pr = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * pr * recall / (pr + recall)
        logger.info("train:     epoch:{0}, pre:{1:.3f}, recall:{2:.3f}, f1:{3:.3f}, loss:{4:.3f}".format(epoch, pr, recall, f1, total_loss / len(train_load_pair)))
        model.eval()
        # valid
        total_loss, total_tp, total_fp, total_fn = 0.0, 0.0, 0.0, 0.0
        for batch in valid_load_pair:
            batch_src_images, batch_tgt_images, batch_label = batch
            batch_src_images, batch_tgt_images, batch_label = batch_src_images.to(device), batch_tgt_images.to(device), batch_label.to(device)
            
            with torch.no_grad():
                with autocast():
                    src_fea = model(batch_src_images)
                    tgt_fea = model(batch_tgt_images)
                    loss = cosent.loss_function(src_fea, tgt_fea, batch_label)
            total_loss += loss
            cos_sim = torch.cosine_similarity(src_fea, tgt_fea)
            cos_sim = (cos_sim + 1) * 0.5
            total_tp += metrics.get_tp(cos_sim, batch_label)
            total_fn += metrics.get_fn(cos_sim, batch_label)
            total_fp += metrics.get_fp(cos_sim, batch_label)
        pr = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * pr * recall / (pr + recall)
        logger.info("valid:     epoch:{0}, pre:{1:.3f}, recall:{2:.3f}, f1:{3:.3f}, loss:{4:.3f}".format(epoch, pr, recall, f1, total_loss / len(valid_load_pair)))
        # test
        total_loss, total_tp, total_fp, total_fn = 0.0, 0.0, 0.0, 0.0
        for batch in test_load_pair:
            batch_src_images, batch_tgt_images, batch_label = batch
            batch_src_images, batch_tgt_images, batch_label = batch_src_images.to(device), batch_tgt_images.to(device), batch_label.to(device)
            
            with torch.no_grad():
                with autocast():
                    src_fea = model(batch_src_images)
                    tgt_fea = model(batch_tgt_images)
                    loss = cosent.loss_function(src_fea, tgt_fea, batch_label)
            total_loss += loss
            cos_sim = torch.cosine_similarity(src_fea, tgt_fea)
            cos_sim = (cos_sim + 1) * 0.5
            total_tp += metrics.get_tp(cos_sim, batch_label)
            total_fn += metrics.get_fn(cos_sim, batch_label)
            total_fp += metrics.get_fp(cos_sim, batch_label)
        pr = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * pr * recall / (pr + recall)
        logger.info("test:     epoch:{0}, pre:{1:.3f}, recall:{2:.3f}, f1:{3:.3f}, loss:{4:.3f}".format(epoch, pr, recall, f1, total_loss / len(test_load_pair)))
        # if max_f1 <= f1:
        #     max_f1 = f1
        #     torch.save(model.state_dict(), os.path.join(save_path, 'baseline_image_large_512.pkl'))
        #     logger.info("save model success!")
        if epoch > 15:
            torch.save(model.state_dict(), os.path.join(save_path, 'baseline_image_large_512_{0}.pkl'.format(epoch)))
            logger.info("save model success!")
