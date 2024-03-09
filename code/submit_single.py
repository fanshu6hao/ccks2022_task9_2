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
from text.config import set_args_submit_single


# 单个csv文件生成提交的jsonl文件
args = set_args_submit_single()
itemid2emb_text = {}
# text
# 读取embedding文件
with open(args.csv_file) as csv_in_file:
    reader = csv.DictReader(csv_in_file)
    # next(reader)
    for item in tqdm(reader):
        src_fea, tgt_fea = "", ""
        for src_substr in item['src_item_emb'][1:-1].split():
            src_fea += f"{src_substr},"
        for tgt_substr in item['tgt_item_emb'][1:-1].split():
            tgt_fea += f"{tgt_substr},"
        itemid2emb_text[str(item['src_item_id'])] = '[' + src_fea[:-1] + ']'
        itemid2emb_text[str(item['tgt_item_id'])] = '[' + tgt_fea[:-1] + ']'

print(len(itemid2emb_text))
# len(eval(itemid2emb_text['101c981131135096efce9ddd3b93a545']))


test_data = pd.read_csv(args.test_info)

columns2 = ['src_item_id', 'tgt_item_id', 'item_label']

# 根据embedding计算相似度 
def compute(item_emb_1: List[float], item_emb_2: List[float]) -> float:
    return (dot(item_emb_1, item_emb_2) / (norm(item_emb_1) * norm(item_emb_2)) + 1) * 0.5


id2cate = {}
for index, row in test_data.iterrows():
    id2cate[str(row['item_id'])] = row['cate_name']
    
    
# 正确应该输出 20707
cate_Flag = False # True表示将cate不同的阈值设为0.99
num = 0
n, n1 = 0, 0
np, nf = 0, 0
update_num_p1, update_num_p2, update_num_f1, update_num_f2  = 0, 0, 0, 0
file_name = args.save_file_name
f_out = open(file_name, encoding='utf-8', mode='w')
with open(args.test_pair, encoding='utf-8', mode='r') as f:
    for line in tqdm(f.readlines()):
        try:
            num += 1
            line = line.strip()
            item = json.loads(line)
            src_item_id = item['src_item_id']
            tgt_item_id = item['tgt_item_id']
            src_item_emb = itemid2emb_text[src_item_id]
            tgt_item_emb = itemid2emb_text[tgt_item_id]
            
            
            cate1 = id2cate[src_item_id]
            cate2 = id2cate[tgt_item_id]
            
            total_sim = compute(eval(src_item_emb), eval(tgt_item_emb))
            
            threshold = args.threshold
            t = threshold
            
            if cate1 != cate2 and total_sim > threshold: # 去除cate_name不同的
                if cate_Flag:
                    threshold = 0.99
                n += 1
                
            if total_sim > threshold:
                np += 1
            else:
                nf += 1
            
            # threshold = (itemid2emb_text[src_item_id+tgt_item_id] + itemid2emb_pic[src_item_id+tgt_item_id]) / 2.0
            out_item = {"src_item_id": src_item_id, "src_item_emb": src_item_emb, "tgt_item_id": tgt_item_id, "tgt_item_emb": tgt_item_emb, "threshold": threshold}
            f_out.write(json.dumps(out_item) + '\n')
        except Exception as e:
            print(e)
            print(line)

f_out.close()
print("是否去除cate不同的：", cate_Flag)
print("threshold:", t)
print("cate不同：", n)
if cate_Flag:
    print("正样本：", np)
else:
    print("正样本：", np-n)
# print("负样本：", nf)
# num