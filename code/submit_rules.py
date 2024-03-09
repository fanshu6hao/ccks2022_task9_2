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
import re

from text.config import set_args_submit


# 生成提交的jsonl文件，并加上规则修改阈值

args = set_args_submit()

itemid2emb_text = {}
# text
# 读取embedding文件
csv_text = args.csv_text
with open(csv_text) as csv_in_file:
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

# print(len(eval(itemid2emb_text['f38538a9261df6d4b6ae0e7ebb46c0ad'])))


# pic
# 读取embedding文件
itemid2emb_pic = {}
csv_pic = args.csv_pic
with open(csv_pic) as csv_in_file:
    reader = csv.DictReader(csv_in_file)
    # next(reader)
    for item in tqdm(reader):
        src_fea, tgt_fea = "", ""
        for src_substr in item['src_item_emb'][1:-1].split():
            src_fea += f"{src_substr},"
        for tgt_substr in item['tgt_item_emb'][1:-1].split():
            tgt_fea += f"{tgt_substr},"
        
        itemid2emb_pic[str(item['src_item_id'])] = '[' + src_fea[:-1] + ']'
        itemid2emb_pic[str(item['tgt_item_id'])] = '[' + tgt_fea[:-1] + ']'

print(len(itemid2emb_pic))

# print(len(eval(itemid2emb_pic['f38538a9261df6d4b6ae0e7ebb46c0ad'])))


test_data = pd.read_csv(args.test_info)

# 对数据进行 norm，再concat
def norm_concat(src1, src2, tgt1, tgt2):
    src1, src2 = torch.tensor(src1), torch.tensor(src2)
    tgt1, tgt2 = torch.tensor(tgt1), torch.tensor(tgt2)
    src_emb = torch.cat([F.normalize(src1, dim=0), F.normalize(src2, dim=0)], axis=0)
    tgt_emb = torch.cat([F.normalize(tgt1, dim=0), F.normalize(tgt2, dim=0)], axis=0)
    return src_emb.tolist(), tgt_emb.tolist()


# 根据embedding计算相似度 
def compute(item_emb_1: List[float], item_emb_2: List[float]) -> float:
    return (dot(item_emb_1, item_emb_2) / (norm(item_emb_1) * norm(item_emb_2)) + 1) * 0.5


error_list = ['other/其他', 'OTHER', 'other其他', '其他/other', '其他other', '见描述', '其他品牌', '其他', 'nan', '']
# 相同brand返回True
def same_brands(v1, v2):
    # 第10列是 brand
    v1 = v1['brand'].values[0]
    v2 = v2['brand'].values[0]
    v1 = str(v1)
    v2 = str(v2)
    if v1 not in error_list and v1 == v2:
        return True
    else:
        return False
    
    
# 不同 brand 返回True
def diff_brands(v1, v2):
    # 第10列是 brand
    #v1 = test_data.query('item_id==@src_id').values[0][10]
    #v2 = test_data.query('item_id==@tgt_id').values[0][10]
    v1 = v1['brand'].values[0]
    v2 = v2['brand'].values[0]
    v1 = str(v1)
    v2 = str(v2)
    if v1 not in error_list and v2 not in error_list and v1 != v2:
        return True
    else:
        return False

    
# 型号 不同返回 True
def diff_xinghao(v1, v2):
    # 第12列是 xinghao
    #v1 = test_data.query('item_id==@src_id').values[0][12]
    # v2 = test_data.query('item_id==@tgt_id').values[0][12]
    v1 = v1['xinghao'].values[0]
    v2 = v2['xinghao'].values[0]
    v1 = str(v1).lower()
    v2 = str(v2).lower()
    if v1 not in error_list and v2 not in error_list and (v1 != v2):
        return True
    return False    
    

# 型号 相同返回 True
def same_xinghao(v1, v2):
    # 第12列是 xinghao
    #v1 = test_data.query('item_id==@src_id').values[0][12]
    #v2 = test_data.query('item_id==@tgt_id').values[0][12]
    v1 = v1['xinghao'].values[0]
    v2 = v2['xinghao'].values[0]
    v1 = str(v1).lower()
    v2 = str(v2).lower()
    if v1 not in error_list and v2 not in error_list and (v1 == v2):
        return True
    return False


def same_huohao(v1, v2):
    v1 = v1['huohao'].values[0]
    v2 = v2['huohao'].values[0]
    v1 = str(v1).lower()
    v2 = str(v2).lower()
    if v1 not in error_list and v2 not in error_list:
        if v1 in v2 or v2 in v1:
            return True


def is_all_chinese(strs):
    for i in strs:
        if not '\u4e00' <= i <= '\u9fa5':
            return False
    return True


# 型号、货号 清洗再筛选？
def clean_xinghao(df):
    xinghao = str(df['xinghao'])
    # 如果 型号 纯数字、纯汉字，改为空
    if str(xinghao).isdigit() or is_all_chinese(str(xinghao)):
        xinghao = ''
    return xinghao


def clean_huohao(df):
    huohao = str(df['huohao'])
    if str(huohao).isdigit() and len(str(huohao)) < 5:
        huohao = ''
    return huohao


test_data.loc[:, 'xinghao'] = test_data.apply(lambda x:clean_xinghao(x), axis=1)
test_data.loc[:, 'huohao'] = test_data.apply(lambda x:clean_huohao(x), axis=1)


# 比较 “投资贵金属”
def compare_tzgjs(value1, value2):
    # 返回值 1 表示 认为两值相同
    # 返回值 0 表示 不确定
    # 返回值 -1 表示 不相同
    len1 = len(value1)
    len2 = len(value2)
    if value1 and value2:
        if len1 == 1:
            if value1[0] in value2:
                return 1
            else:
                return -1
        elif len2 == 1:
            if value2[0] in value1:
                return 1
            else:
                return -1
        else:
            if value1 == value2:
                return 1
            return 0 # 两个长度都不为1，不好比较
    else:
        return 0


def extract_values_tzgjs(v1, v2):
    # 返回值 1 表示 相同
    # 返回值 0 表示 不确定
    # 返回值 -1 表示 不相同
    title1 = v1['title'].values[0]
    title2 = v2['title'].values[0]
    # 生肖
    sx1 = re.findall(r'鼠|牛|虎|兔|龙|蛇|马|羊|猴|鸡|狗|猪', title1)
    sx2 = re.findall(r'鼠|牛|虎|兔|龙|蛇|马|羊|猴|鸡|狗|猪', title2)
    sx1 = list(set(sx1))
    sx2 = list(set(sx2))
    pd1 = compare_tzgjs(sx1, sx2)

    # 年份
    year1 = re.findall(r'(\d{4})+年', title1)
    year2 = re.findall(r'(\d{4})+年', title2)
    if not year1:
        year1 = re.findall(r'(\d{4})', title1)
    if not year2:
        year2 = re.findall(r'(\d{4})', title2)
    year1 = list(set(year1))
    year2 = list(set(year2))
    pd2 = compare_tzgjs(year1, year2)

    # 面额
    # 去掉长度大于3的？
    money1 = re.findall(r'(\d+)+元', title1)
    money2 = re.findall(r'(\d+)+元', title2)
    money1 = list(set(money1))
    money2 = list(set(money2))
    pd3 = compare_tzgjs(money1, money2)

    if pd1 == -1 or pd2 == -1:
        return -1
    elif pd1 == 1 or pd2 == 1:
        # 年份相同，面额不同
        if pd2 == 1 and pd3 == -1:
            return -1
        return 1
    elif pd3 == -1:
        return -1
    return 0



# 不等于、且相互不包含
def diff_values(v1, v2):
    v1 = str(v1)
    v2 = str(v2)
    if v1 != v2 and (v1 not in v2) and (v2 not in v1):
        return True
    return False


def same_values(v1, v2):
    if v1 and v2:
        v1 = str(v1)
        v2 = str(v2)
        if v1 in v2 or v2 in v1:
            return True

# 洗烘套装、洗衣机
# 抽出 烘干机型号、洗衣机型号
def extract_values_xhtz(v1, v2):
    itempvs1 = v1['item_pvs'].values[0]
    itempvs2 = v2['item_pvs'].values[0]
    hgj1 = re.findall(r'烘干机型号:(.*?);', itempvs1)
    hgj2 = re.findall(r'烘干机型号:(.*?);', itempvs2)
    xyj1 = re.findall(r'洗衣机型号:(.*?);', itempvs1)
    xyj2 = re.findall(r'洗衣机型号:(.*?);', itempvs2)
    # 去掉中文的
    if is_all_chinese(hgj1) or is_all_chinese(hgj2) or is_all_chinese(xyj1) or is_all_chinese(xyj2):
        return None, None, None, None
    pd1 = same_values(hgj1, hgj2)
    pd11 = same_values(xyj1, xyj2)
    pd2 = diff_values(hgj1, hgj2)
    pd22 = diff_values(xyj1, xyj2)
    return pd1, pd11, pd2, pd22
    
    
    
# valid 正确应该输出 20707
# test 正确应该输出 15909
Flag_Norm = True  # 是否先norm，再将 text 拼接 pic
print("是否先NORM，再拼接：", Flag_Norm)
Flag_update_threshold = args.add_rules
print("是否更改threshold：", Flag_update_threshold)
num = 0
n = 0
np, nf = 0, 0
update_num_tz_p, update_num_tz_f  = 0, 0
update_num_xhtz_p, update_num_xhtz_f = 0, 0
update_num_p1, update_num_p2, update_num_f1, update_num_f2, update_num_f3 = 0, 0, 0, 0, 0
save_file_name = args.save_file_name
f_out = open(save_file_name, encoding='utf-8', mode='w')
with open(args.test_pair, encoding='utf-8', mode='r') as f:
    for line in tqdm(f.readlines()):
        try:
            num += 1
            line = line.strip()
            item = json.loads(line)
            src_item_id = item['src_item_id']
            tgt_item_id = item['tgt_item_id']
            
            threshold = args.threshold
            t = threshold
        
            v1 = test_data.query('item_id==@src_item_id')
            v2 = test_data.query('item_id==@tgt_item_id')

            cate1 = v1['cate_name'].values[0]
            cate2 = v2['cate_name'].values[0]
            
            
            if Flag_Norm:  # 先 norm 再 concat
                src1, tgt1 = eval(itemid2emb_text[src_item_id]), eval(itemid2emb_text[tgt_item_id])
                src2, tgt2 = eval(itemid2emb_pic[src_item_id]), eval(itemid2emb_pic[tgt_item_id])
                
                src_emb, tgt_emb = norm_concat(src1, src2, tgt1, tgt2)
                text_sim = compute(src1, tgt1)
                pic_sim = compute(src2, tgt2)
                total_sim = compute(src_emb, tgt_emb)
                
                if cate1 == '洗衣机':
                    cate1 = "洗烘套装"
                if cate2 == '洗衣机':
                    cate2 = "洗烘套装"
                
                if cate1 != cate2 and total_sim > threshold: # 统计cate_name不同的
                    # threshold = 0.82
                    n += 1
                
                #####################################################
                ###################### 人工规则 ######################
                #####################################################
                else:
                    if Flag_update_threshold:
                        if cate1 == '投资贵金属' and cate2 == '投资贵金属':
                            if (0.95 > total_sim > threshold) and extract_values_tzgjs(v1, v2) == -1: 
                                threshold = 0.95
                                update_num_tz_p += 1
                            if 0.62 < total_sim < threshold and extract_values_tzgjs(v1, v2) == 1:
                                threshold = 0.62
                                update_num_tz_f += 1
                        elif cate1 == "洗烘套装" and cate2 == "洗烘套装":
                            pd1, pd11, pd2, pd22 = extract_values_xhtz(v1, v2)
                            # 洗衣机型号、烘干机型号 有一个相同的就降低阈值
                            if total_sim < threshold and (pd1 or pd11):
                                threshold = 0.7
                                update_num_xhtz_p += 1
                            if total_sim > threshold:
                                # 品牌不同的 或者 洗衣机型号、烘干机型号都不同，提高阈值
                                if diff_brands(v1, v2) or (pd2 and pd22):
                                    threshold = 0.82
                                    update_num_xhtz_f += 1
                        else:
                            # 如果text相似度高，且 brand 相同，或者相似度极高，阈值降低
                            # if (text_sim > 0.92 and same_brands(src_item_id, tgt_item_id)) or text_sim > 0.95 or pic_sim > 0.905:
                            if (text_sim > 0.92 and same_brands(v1, v2)) or pic_sim > 0.905:
                                if total_sim < threshold:
                                    threshold = 0.7
                                    update_num_f1 += 1

                            # 如果 text 和 pic 的相似度都不是很高，且 brand 不同，提高阈值
                            if text_sim < (t+0.02) and pic_sim < (t+0.02) and diff_brands(v1, v2):
                                if t+0.02 > total_sim > threshold:
                                    threshold = t+0.02
                                    update_num_p1 += 1


                            # 型号相同的，降低阈值
                            if 0.7 < total_sim < threshold and same_brands(v1, v2):
                                if same_xinghao(v1, v2):
                                    threshold = 0.7
                                    update_num_f2 += 1
                            
                            # 货号相同的，降低阈值
                            if 0.7 < total_sim < threshold and same_huohao(v1, v2):
                                threshold = 0.7
                                update_num_f3 += 1


                            # 品牌不同，型号不同，pic相似度不是很高，提高阈值
                            if pic_sim < (t+0.02) and 0.83 > total_sim > threshold:
                                if diff_brands(v1, v2) and diff_xinghao(v1, v2):
                                    threshold = 0.83
                                    update_num_p2 += 1

                    
                if total_sim > threshold:
                    np += 1
                else:
                    nf += 1
                
                
                src_item_emb = str(src_emb)
                tgt_item_emb = str(tgt_emb)
            else:  # 直接concat
                src_item_emb = itemid2emb_text[src_item_id][:-1] + ',' + itemid2emb_pic[src_item_id][1:]
                tgt_item_emb = itemid2emb_text[tgt_item_id][:-1] + ',' + itemid2emb_pic[tgt_item_id][1:]
            
            # threshold = (itemid2emb_text[src_item_id+tgt_item_id] + itemid2emb_pic[src_item_id+tgt_item_id]) / 2.0
            out_item = {"src_item_id": src_item_id, "src_item_emb": src_item_emb, "tgt_item_id": tgt_item_id, "tgt_item_emb": tgt_item_emb, "threshold": threshold}
            f_out.write(json.dumps(out_item) + '\n')
        except Exception as e:
            print(e)
            print(line)

f_out.close()
print("threshold:", t)
print("cate不同：", n)
print("正样本：", np-n)
if Flag_update_threshold:
    print('修改 “投资贵金属” 阈值数据条数（提高、降低）:', update_num_tz_p, update_num_tz_f)
    print('修改 “洗烘套装” 阈值数据条数（提高、降低）:', update_num_xhtz_p, update_num_xhtz_f)
    print('提高阈值数据条数：', update_num_p1, update_num_p2)
    print('降低阈值数据条数：', update_num_f1, update_num_f2, update_num_f3)
    p = update_num_tz_p + update_num_xhtz_p + update_num_p1 + update_num_p2
    f =  update_num_tz_f +  update_num_xhtz_f + update_num_f1 + update_num_f2 + update_num_f3
    print('\n共提高条数：', p, '\n共降低条数：', f)
num