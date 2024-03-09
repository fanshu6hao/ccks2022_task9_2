import os
import csv
import pandas as pd
import json
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import sys
import re

from model import Baseline
from config import set_args_predict


np.set_printoptions(threshold=sys.maxsize)


# # 清洗“brand”
# def brand_clean(brand):
#     error_brand = ['other/其他', 'OTHER', 'other其他', '其他/other', '其他other', '其他']
#     if brand in error_brand:
#         brand = '其他品牌'
#     return brand


# 清洗“brand”
def brand_clean(brand):
    error_brand = ['other/其他', 'OTHER', 'other其他', '其他/other', '其他other', '见描述', '其他品牌']
    if brand in error_brand:
        brand = '其他'
    return brand


# # 清洗“brand”
# def brand_clean(brand):
#     error_brand = ['other/其他', 'OTHER', 'other其他', '其他/other', '其他other', '见描述', '其他品牌', '属性见描述']
#     if brand in error_brand:
#         brand = '其他'
#     return brand


class ItemDataset(Dataset):

    def __init__(self, data, tokenizer, max_len_main, max_len_secondary, max_len_full):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len_main = max_len_main
        self.max_len_secondary = max_len_secondary
        self.max_len_full = max_len_full

    def __len__(self):
        return len(self.data)

    def token(self, item):
        CLS, SEP = '[CLS]', '[SEP]'
        if "item_pvs" not in item.keys():
            item['item_pvs'] = ""
        item_id, industry_name, cate_id, cate_name, cate_id_path, cate_name_path, item_image_name, title, item_pvs, sku_pvs, brand = \
        item['item_id'], item['industry_name'], item[
            'cate_id'], item['cate_name'], item['cate_id_path'], item['cate_name_path'], item['item_image_name'], item[
            'title'], item['item_pvs'], item['sku_pvs'], item['brand']
    
        brand = brand_clean(brand)
        # sku_pvs = sku_pvs.replace('#', '')
        # sku_pvs = sku_pvs.replace(' ', '')
        # sku_pvs = re.sub('#| |（|）|★|【|】|\(|\)|\+|/|\[|\]|\{|\}|&|\^(2|3)|°|\$|@|!|！|✅|￥|？|《|》|☆', '', sku_pvs) # 去除一些符号
        
        context_main = f"无"
        cotext_secondary = f"无"
        context_full = f"{industry_name}*{cate_name}*品牌:{brand}*{title}*{item_pvs}*{sku_pvs}"
        
        token_context_main = self.tokenizer.encode_plus(context_main, max_length=self.max_len_main, truncation=True,
                                                        add_special_tokens=True, padding="max_length")
        token_context_secondary = self.tokenizer.encode_plus(cotext_secondary,
                                                             max_length=self.max_len_secondary,
                                                             truncation=True,
                                                             add_special_tokens=True,
                                                             padding="max_length")
        token_context_full = self.tokenizer.encode_plus(context_full, max_length=self.max_len_full, truncation=True,
                                                        add_special_tokens=True, padding="max_length")
        return item_id, token_context_main, token_context_secondary, token_context_full

    def collate(self, examples):
        batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full = [], [], [], []
        for item in examples:
            item_id, token_context_main, token_context_secondary, token_context_full = self.token(item)
            batch_item_ids.append(item_id)
            batch_token_context_main.append(token_context_main)
            batch_token_context_secondary.append(token_context_secondary)
            batch_token_context_full.append(token_context_full)
        return batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full

    def __getitem__(self, index):
        return self.data[index]
    

class PairDataset(Dataset):

    def __init__(self, data, item_token_dict):
        self.data = data
        self.item_token_dict = item_token_dict

    def __len__(self):
        return len(self.data)

    def collate(self, examples):
        batch_src_input_ids_main, batch_src_segment_ids_main, batch_src_attention_mask_main = [], [], []
        batch_src_input_ids_secondary, batch_src_segment_ids_secondary, batch_src_attention_mask_secondary = [], [], []
        batch_src_input_ids_full, batch_src_segment_ids_full, batch_src_attention_mask_full = [], [], []
        batch_tgt_input_ids_main, batch_tgt_segment_ids_main, batch_tgt_attention_mask_main = [], [], []
        batch_tgt_input_ids_secondary, batch_tgt_segment_ids_secondary, batch_tgt_attention_mask_secondary = [], [], []
        batch_tgt_input_ids_full, batch_tgt_segment_ids_full, batch_tgt_attention_mask_full = [], [], []
        batch_src_item_ids = []
        batch_tgt_item_ids = []

        for item in examples:
            batch_src_input_ids_main.append(
                self.item_token_dict[item['src_item_id']]["token_context_main"]["input_ids"])
            batch_src_segment_ids_main.append(
                self.item_token_dict[item['src_item_id']]["token_context_main"]["token_type_ids"])
            batch_src_attention_mask_main.append(
                self.item_token_dict[item['src_item_id']]["token_context_main"]["attention_mask"])

            batch_src_input_ids_secondary.append(
                self.item_token_dict[item['src_item_id']]["token_context_secondary"]["input_ids"])
            batch_src_segment_ids_secondary.append(
                self.item_token_dict[item['src_item_id']]["token_context_secondary"]["token_type_ids"])
            batch_src_attention_mask_secondary.append(
                self.item_token_dict[item['src_item_id']]["token_context_secondary"]["attention_mask"])

            batch_src_input_ids_full.append(
                self.item_token_dict[item['src_item_id']]["token_context_full"]["input_ids"])
            batch_src_segment_ids_full.append(
                self.item_token_dict[item['src_item_id']]["token_context_full"]["token_type_ids"])
            batch_src_attention_mask_full.append(
                self.item_token_dict[item['src_item_id']]["token_context_full"]["attention_mask"])

            batch_tgt_input_ids_main.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_main"]["input_ids"])
            batch_tgt_segment_ids_main.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_main"]["token_type_ids"])
            batch_tgt_attention_mask_main.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_main"]["attention_mask"])

            batch_tgt_input_ids_secondary.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_secondary"]["input_ids"])
            batch_tgt_segment_ids_secondary.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_secondary"]["token_type_ids"])
            batch_tgt_attention_mask_secondary.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_secondary"]["attention_mask"])

            batch_tgt_input_ids_full.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_full"]["input_ids"])
            batch_tgt_segment_ids_full.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_full"]["token_type_ids"])
            batch_tgt_attention_mask_full.append(
                self.item_token_dict[item['tgt_item_id']]["token_context_full"]["attention_mask"])

            batch_src_item_ids.append(item['src_item_id'])
            batch_tgt_item_ids.append(item['tgt_item_id'])

        batch_src_token_main = {
            "input_ids": torch.tensor(batch_src_input_ids_main),
            "segment_ids": torch.tensor(batch_src_segment_ids_main),
            "attention_mask": torch.tensor(batch_src_attention_mask_main)
        }
        batch_src_token_secondary = {
            "input_ids": torch.tensor(batch_src_input_ids_secondary),
            "segment_ids": torch.tensor(batch_src_segment_ids_secondary),
            "attention_mask": torch.tensor(batch_src_attention_mask_secondary)
        }
        batch_src_token_full = {
            "input_ids": torch.tensor(batch_src_input_ids_full),
            "segment_ids": torch.tensor(batch_src_segment_ids_full),
            "attention_mask": torch.tensor(batch_src_attention_mask_full)
        }
        batch_tgt_token_main = {
            "input_ids": torch.tensor(batch_tgt_input_ids_main),
            "segment_ids": torch.tensor(batch_tgt_segment_ids_main),
            "attention_mask": torch.tensor(batch_tgt_attention_mask_main)
        }
        batch_tgt_token_secondary = {
            "input_ids": torch.tensor(batch_tgt_input_ids_secondary),
            "segment_ids": torch.tensor(batch_tgt_segment_ids_secondary),
            "attention_mask": torch.tensor(batch_tgt_attention_mask_secondary)
        }
        batch_tgt_token_full = {
            "input_ids": torch.tensor(batch_tgt_input_ids_full),
            "segment_ids": torch.tensor(batch_tgt_segment_ids_full),
            "attention_mask": torch.tensor(batch_tgt_attention_mask_full)
        }
        return batch_src_item_ids, batch_tgt_item_ids, batch_src_token_main, batch_src_token_secondary, batch_src_token_full, batch_tgt_token_main, batch_tgt_token_secondary, batch_tgt_token_full

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    args = set_args_predict()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    device = torch.device('cuda:0')
    max_len_main = 1
    max_len_secondary = 1
    max_len_full = 256
    # 12层模型 bs 128  3721
    batch_size = args.batch_size
    
    threshold = args.threshold

    datafolder = args.datafolder
    save_path = args.save_path
    model_path = args.model_path
    load_model_name = args.load_model_name + '.pkl'
    save_csv_name = args.save_csv_name + '.csv'
    
    encoder_type = args.encoder_type  # "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
    output_dim = args.output_dim
    # roberta、roberta_L12、roberta_wwm、roberta_wwm_L12、simbert
    bert_model_path = args.bert_model_path
    # token_file = 'item_test_token.json'
    # ReWrite_Token = True  # False表示不重新生成Token文件
    
    print("Model path :", bert_model_path)

    with open(os.path.join(datafolder, 'item_test_info.json'), 'r') as file:
        item_test_info = json.load(file)
    with open(os.path.join(datafolder, 'item_test_pair.json'), 'r') as file:
        item_test_pair = json.load(file)

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

    test_data = ItemDataset(item_test_info, tokenizer, max_len_main, max_len_secondary, max_len_full)
    test_load_data = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.collate, shuffle=True, num_workers=4)

    item_token_dict = {}
    for batch in tqdm(test_load_data):
        batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full = batch
        for item_id, token_context_main, token_context_secondary, token_context_full in zip(batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full):
            item_token_dict[item_id] = {
                "token_context_main": token_context_main,
                "token_context_secondary": token_context_secondary,
                "token_context_full": token_context_full
            }
    # if not os.path.exists(os.path.join(datafolder, token_file)) or ReWrite_Token == True:
    #     test_item_token_list, test_item_token = [], []
    #     for batch in tqdm(test_load_data):
    #         batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full = batch
    #         test_item_token_list.extend({"item_id": item_id, "token_context_main": dict(token_context_main), "token_context_secondary": dict(token_context_secondary), "token_context_full": dict(token_context_full)} for item_id, token_context_main, token_context_secondary, token_context_full in zip(batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full))

    #     with open(os.path.join(datafolder, token_file), 'w') as f:
    #         for item in test_item_token_list:
    #             item = json.dumps(item)
    #             test_item_token.append(item)
    #         json.dump(test_item_token, f, indent=4, ensure_ascii=False)

    # with open(os.path.join(datafolder, token_file), 'r') as f:
    #     item_test_list = json.load(f)
    #     for item in item_test_list:
    #         item = json.loads(item)
    #         item_token_dict[item["item_id"]] = {
    #             "token_context_main": item["token_context_main"],
    #             "token_context_secondary": item["token_context_secondary"],
    #             "token_context_full": item["token_context_full"]
    #         }

    test_pair = PairDataset(item_test_pair, item_token_dict)
    test_load_pair = DataLoader(test_pair, batch_size=batch_size, collate_fn=test_pair.collate, num_workers=4)

    encoder = BertModel.from_pretrained(bert_model_path)
    
    model = Baseline(encoder=encoder, output_dim=output_dim, encoder_type=encoder_type)
    print("encoder_type =", encoder_type)
    print("output_dim =", output_dim)

    # 如果有多张卡，就并行
    # 如果并行训练的话，加载已保存的模型也要并行
    # if torch.cuda.device_count() > 1:
    #     print("\nUse", torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model)

    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, load_model_name)))
    print("Load model {0} success".format(load_model_name))
    model.eval()

    pair_fea_list = []
    for batch in tqdm(test_load_pair):
        batch_src_item_ids, batch_tgt_item_ids, batch_src_token_main, batch_src_token_secondary, batch_src_token_full, batch_tgt_token_main, batch_tgt_token_secondary, batch_tgt_token_full = batch
        batch_src_input_ids, batch_src_segment_ids, batch_src_attention_mask, batch_tgt_input_ids, batch_tgt_segment_ids, batch_tgt_attention_mask = \
            batch_src_token_full[
                "input_ids"].to(device), batch_src_token_full["segment_ids"].to(device), batch_src_token_full[
                "attention_mask"].to(device), batch_tgt_token_full[
                "input_ids"].to(device), batch_tgt_token_full["segment_ids"].to(device), batch_tgt_token_full[
                "attention_mask"].to(device)
        with torch.no_grad():
            src_fea = model(batch_src_input_ids, batch_src_attention_mask, batch_src_segment_ids)
            tgt_fea = model(batch_tgt_input_ids, batch_tgt_attention_mask, batch_tgt_segment_ids)

        pair_fea_list.extend([src_item_id, src.detach().cpu().numpy(), tgt_item_id, tgt.detach().cpu().numpy(), threshold] for src_item_id, tgt_item_id, src, tgt in zip(batch_src_item_ids, batch_tgt_item_ids, src_fea, tgt_fea))

    result = pd.DataFrame(columns=["src_item_id", "src_item_emb", "tgt_item_id", "tgt_item_emb", "threshold"], data=pair_fea_list)
    result.to_csv(os.path.join(save_path, save_csv_name), index=False)
    print("predict and save {0} success!".format(save_csv_name))
