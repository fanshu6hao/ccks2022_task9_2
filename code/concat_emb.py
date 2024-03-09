import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import csv
from typing import List
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from csv import DictWriter

from text.config import set_args_concat


# 将两个csv结果文件合并
args = set_args_concat()

itemid2emb_1 = {}
# 读取embedding文件
csv1 = args.csv1
with open(csv1) as csv_in_file:
    reader = csv.DictReader(csv_in_file)
    # next(reader)
    for item in tqdm(reader):
        src_fea, tgt_fea = "", ""
        for src_substr in item['src_item_emb'][1:-1].split():
            src_fea += f"{src_substr},"
        for tgt_substr in item['tgt_item_emb'][1:-1].split():
            tgt_fea += f"{tgt_substr},"
        itemid2emb_1[str(item['src_item_id'])] = '[' + src_fea[:-1] + ']'
        itemid2emb_1[str(item['tgt_item_id'])] = '[' + tgt_fea[:-1] + ']'
        
print(len(itemid2emb_1))


itemid2emb_2 = {}
csv2 = args.csv2
with open(csv2) as csv_in_file:
    reader = csv.DictReader(csv_in_file)
    for item in tqdm(reader):
        src_fea, tgt_fea = "", ""
        for src_substr in item['src_item_emb'][1:-1].split():
            src_fea += f"{src_substr},"
        for tgt_substr in item['tgt_item_emb'][1:-1].split():
            tgt_fea += f"{tgt_substr},"
        itemid2emb_2[str(item['src_item_id'])] = '[' + src_fea[:-1] + ']'
        itemid2emb_2[str(item['tgt_item_id'])] = '[' + tgt_fea[:-1] + ']'

print(len(itemid2emb_2))


# 输出维度
# print(len(eval(itemid2emb_1['f38538a9261df6d4b6ae0e7ebb46c0ad'])))
# print(len(eval(itemid2emb_2['f38538a9261df6d4b6ae0e7ebb46c0ad'])))



# 对数据进行 norm，再concat
def norm_concat(src1, src2, tgt1, tgt2):
    src1, src2 = torch.tensor(src1), torch.tensor(src2)
    tgt1, tgt2 = torch.tensor(tgt1), torch.tensor(tgt2)
    src_emb = torch.cat([F.normalize(src1, dim=0), F.normalize(src2, dim=0)], axis=0)
    tgt_emb = torch.cat([F.normalize(tgt1, dim=0), F.normalize(tgt2, dim=0)], axis=0)
    # 转换成numpy格式
    return src_emb.numpy(), tgt_emb.numpy()


num = 0
save_csv = args.save_csv
pair_file = args.test_pair

with open(pair_file, encoding='utf-8', mode='r') as inp, open(save_csv, 'w') as outp:
    writer = DictWriter(outp, fieldnames=['src_item_id', 'src_item_emb', 'tgt_item_id', 'tgt_item_emb'])
    writer.writeheader()
    for line in tqdm(inp.readlines()):
        num += 1
        line = line.strip()
        item = json.loads(line)
        src_item_id = item['src_item_id']
        tgt_item_id = item['tgt_item_id']

        src_1, src_2 = eval(itemid2emb_1[src_item_id]), eval(itemid2emb_2[src_item_id])
        tgt_1, tgt_2 = eval(itemid2emb_1[tgt_item_id]), eval(itemid2emb_2[tgt_item_id])
        src1, tgt1 = norm_concat(src_1, src_2, tgt_1, tgt_2)
        
        writer.writerow({"src_item_id": src_item_id, "src_item_emb": str(src1), "tgt_item_id": tgt_item_id, "tgt_item_emb": str(tgt1)})
num

print('合并两个csv文件完成！')