import os
import random
import json
from sched import scheduler
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time
import heapq
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from config import set_args_train
from model import Baseline
import re



ReWrite_Token = False # False表示不重新生成Token文件


# 清洗“brand”
def brand_clean(brand):
    error_brand = ['other/其他', 'OTHER', 'other其他', '其他/other', '其他other', '见描述', '其他品牌', '属性见描述']
    if brand in error_brand:
        brand = '其他'
    return brand


# 清洗“brand”
def brand_clean_new(brand):
    error_brand = ['other/其他', 'OTHER', 'other其他', '其他/other', '其他other', '其他', '见描述', '属性见描述', '其他品牌']
    if brand in error_brand:
        brand = ''
    return brand


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
    
        if item_id == '916417ac1f22481f37b295fba5db70a6':
            industry_name = '消费电子'
        
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
        batch_label = []
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

            batch_label.append(int(item['item_label']))
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
        return batch_src_item_ids, batch_tgt_item_ids, batch_src_token_main, batch_src_token_secondary, batch_src_token_full, batch_tgt_token_main, batch_tgt_token_secondary, batch_tgt_token_full, torch.tensor(
            batch_label).float()

    def __getitem__(self, index):
        return self.data[index]
        

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
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
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

    def get_acc(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, self.beta).float()
        correct = (y_pred == y_true).sum()
        return correct.item()


def train(model, train_load_pair, optimizer, scheduler, cosent, metrics, epoch):
    model.train()
    torch.cuda.synchronize()
    since = time.time()
    total_label, total_pred = [], []
    total_loss, total_acc, total_tp, total_fp, total_fn, data_num, batch_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
    # train
    for batch in train_load_pair:
        _, _, batch_src_token_main, batch_src_token_secondary, batch_src_token_full, batch_tgt_token_main, batch_tgt_token_secondary, batch_tgt_token_full, batch_label = batch
        batch_src_input_ids, batch_src_segment_ids, batch_src_attention_mask, batch_tgt_input_ids, batch_tgt_segment_ids, batch_tgt_attention_mask, batch_label = \
            batch_src_token_full[
                "input_ids"].to(device), batch_src_token_full["segment_ids"].to(device), batch_src_token_full[
                "attention_mask"].to(device), batch_tgt_token_full[
                "input_ids"].to(device), batch_tgt_token_full["segment_ids"].to(device), batch_tgt_token_full[
                "attention_mask"].to(device), batch_label.to(device)
        data_num += len(batch_label)  # 统计数据总长度，用来计算acc
        src_fea = model(batch_src_input_ids, batch_src_attention_mask, batch_src_segment_ids)
        tgt_fea = model(batch_tgt_input_ids, batch_tgt_attention_mask, batch_tgt_segment_ids)
        # cos_sim = F.sigmoid(cos_sim)
        loss = cosent.loss_function(src_fea, tgt_fea, batch_label)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step() # 更新学习率预热参数
        cos_sim = torch.cosine_similarity(src_fea, tgt_fea)
        cos_sim = (cos_sim + 1) * 0.5
        
        pred = cos_sim.cpu().detach().numpy()
        label = batch_label.cpu().detach().numpy()
        total_pred.extend(pred)
        total_label.extend(label)
        # batch_auc += roc_auc_score(label, pred)
        
        total_tp += metrics.get_tp(cos_sim, batch_label)
        total_fn += metrics.get_fn(cos_sim, batch_label)
        total_fp += metrics.get_fp(cos_sim, batch_label)
        total_acc += metrics.get_acc(cos_sim, batch_label)

    pr = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * pr * recall / (pr + recall)
    p_num = total_tp + total_fp  # 预测的正样本数量
    total_auc = roc_auc_score(total_label, total_pred)
    torch.cuda.synchronize()
    time_elapsed = time.time() - since
    print("\n\ntrain:     epoch:{0}, loss:{1:.4f}, acc:{2:.4f}, pre:{3:.4f}, recall:{4:.4f}, f1:{5:.4f}, auc:{6:.4f}, p_num:{7:.0f},"
          "time:{8:.0f}m {9:.0f}s".format(epoch+1, total_loss / len(train_load_pair), total_acc / data_num,
                                          pr, recall, f1, total_auc, p_num, time_elapsed // 60, time_elapsed % 60))

def evaluation(model, data, cosent, metrics, epoch, data_name):
    torch.cuda.synchronize()
    since = time.time()
    model.eval()
    total_label, total_pred = [], []
    src_fea_all, tgt_fea_all, labels = [], [], []
    total_loss, total_acc, total_tp, total_fp, total_fn, data_num, batch_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
    for batch in data:
        _, _, batch_src_token_main, batch_src_token_secondary, batch_src_token_full, batch_tgt_token_main, batch_tgt_token_secondary, batch_tgt_token_full, batch_label = batch
        batch_src_input_ids, batch_src_segment_ids, batch_src_attention_mask, batch_tgt_input_ids, batch_tgt_segment_ids, batch_tgt_attention_mask, batch_label = \
            batch_src_token_full[
                "input_ids"].to(device), batch_src_token_full["segment_ids"].to(device), batch_src_token_full[
                "attention_mask"].to(device), batch_tgt_token_full[
                "input_ids"].to(device), batch_tgt_token_full["segment_ids"].to(device), batch_tgt_token_full[
                "attention_mask"].to(device), batch_label.to(device)

        # torch.cuda.synchronize()
        # since1 = time.time()
        data_num += len(batch_label)  # 统计数据总长度，用来计算acc
        with torch.no_grad():
            src_fea = model(batch_src_input_ids, batch_src_attention_mask, batch_src_segment_ids)
            tgt_fea = model(batch_tgt_input_ids, batch_tgt_attention_mask, batch_tgt_segment_ids)
        # torch.cuda.synchronize()
        # time_elapsed1 = time.time() - since1
        # print("compute1 time:{0:f}m {1:f}s".format(time_elapsed1 // 60, time_elapsed1 % 60))
        src_fea_all.append(src_fea)
        tgt_fea_all.append(tgt_fea)
        labels.append(batch_label)

        loss = cosent.loss_function(src_fea, tgt_fea, batch_label)
        total_loss += loss

        cos_sim = torch.cosine_similarity(src_fea, tgt_fea)
        cos_sim = (cos_sim + 1) * 0.5

        pred = cos_sim.cpu().numpy()
        label = batch_label.cpu().numpy()
        total_pred.extend(pred)
        total_label.extend(label)
        # batch_auc += roc_auc_score(label, pred)
        
        total_tp += metrics.get_tp(cos_sim, batch_label)
        total_fn += metrics.get_fn(cos_sim, batch_label)
        total_fp += metrics.get_fp(cos_sim, batch_label)
        total_acc += metrics.get_acc(cos_sim, batch_label)

    pr = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * pr * recall / (pr + recall)
    p_num = total_tp + total_fp  # 预测的正样本数量
    total_auc = roc_auc_score(total_label, total_pred)
    torch.cuda.synchronize()
    time_elapsed = time.time() - since
    print("----------------------------------------------------------------")
    print(data_name + ":     epoch:{0}, loss:{1:.4f}, acc:{2:.4f}, pre:{3:.4f}, recall:{4:.4f}, f1:{5:.4f}, auc:{6:.4f}, "
                      "p_num:{7:.0f}, time:{8:.0f}m {9:.0f}s".format(epoch+1, total_loss / len(data),
                                                                     total_acc / data_num, pr, recall, f1, total_auc, 
                                                                     p_num, time_elapsed // 60, time_elapsed % 60))
    # 找出最优阈值
    l1, l2, l3, l4, l5 = [], [], [], [], []
    for threshold in np.arange(0.75, 0.83, 0.001):
        results = zip(src_fea_all, tgt_fea_all, labels)
        threshold_t, acc_t, pr_t, recall_t, f1_t = get_evaluation(results, threshold)
        l1.append(threshold_t)
        l2.append(acc_t)
        l3.append(pr_t)
        l4.append(recall_t)
        l5.append(f1_t)
    max_index = l5.index(max(l5))
    print("Best      Threshold:{0:.4f}, acc:{1:.4f}, pre:{2:.4f}, recall:{3:.4f}, f1:{4:.4f}".format(l1[max_index], l2[max_index],
                                                                                       l3[max_index], l4[max_index], l5[max_index]))
    max_three_f1 = heapq.nlargest(3, l5)
    threshold_for_maxf1 = [l1[l5.index(item)] for item in max_three_f1]
    max_three_f1 = [item.item() for item in max_three_f1]
    print('Top 3 F1 : {0:.4f}, {1:.4f}, {2:.4f}'.format(max_three_f1[0], max_three_f1[1], max_three_f1[2]), f'; threshold : {threshold_for_maxf1}')

    return f1, total_auc, l5[max_index].item(), l1[max_index].item()


def get_evaluation(results, threshold):
    total_acc, total_tp, total_fp, total_fn, data_num = 0.0, 0.0, 0.0, 0.0, 0
    metrics = MetricsCalculator(beta=threshold)
    for src_fea, tgt_fea, batch_label in results:
        data_num += len(batch_label)  # 统计数据总长度，用来计算acc
        cos_sim = torch.cosine_similarity(src_fea, tgt_fea)
        cos_sim = (cos_sim + 1) * 0.5
        total_tp += metrics.get_tp(cos_sim, batch_label)
        total_fn += metrics.get_fn(cos_sim, batch_label)
        total_fp += metrics.get_fp(cos_sim, batch_label)
        total_acc += metrics.get_acc(cos_sim, batch_label)

    pr = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * pr * recall / (pr + recall)

    # print("Threshold:{0:.4f}, acc:{1:.4f}, pre:{2:.4f}, recall:{3:.4f}, f1:{4:.4f}".format(threshold, total_acc / data_num,
    #                                                                                    pr, recall, f1))
    return threshold, total_acc / data_num, pr, recall, f1


''' 对list格式的dict进行去重'''
def remove_list_dict_duplicate(list_dict_data):
    return [dict(t) for t in {tuple(d.items()) for d in list_dict_data}]


# 统计正负样本数量
def count_label(data):
    return sum(eval(item['item_label']) for item in data)


# 构造dataloader
def Get_Dataloader(NoTest, Augment, datafolder, bert_model_path, token_file, batch_size, max_len_main, max_len_secondary, max_len_full):
    # sourcery skip: low-code-quality
    with open(os.path.join(datafolder, 'item_train_info.json'), 'r') as file:
        item_train_info = json.load(file)
    with open(os.path.join(datafolder, 'item_train_pair.json'), 'r') as file:
        item_train_pair = json.load(file)

    # key 为 id，value 为 cate_name
    id2cate = {}
    for item in item_train_info:
        id2cate[str(item['item_id'])] = item['cate_name']
        
    # # cate_name不同的丢掉
    # print("~~~去除cate_name不同的pair~~~")
    # for line in item_train_pair:
    #     if line['item_label'] == '1':
    #         if id2cate[line['src_item_id']] != id2cate[line['tgt_item_id']]:
    #             item_train_pair.remove(line)

    random.seed(2022)
    random.shuffle(item_train_pair)
    train_pair = item_train_pair[:int(0.8 * len(item_train_pair))]
    valid_pair = item_train_pair[int(0.8 * len(item_train_pair)):int(0.9 * len(item_train_pair))]
    # valid_pair = item_train_pair[int(0.8 * len(item_train_pair))]
    test_pair = item_train_pair[int(0.9 * len(item_train_pair)):]
    if NoTest:
        train_pair.extend(test_pair)  # 将test也加入train
    print("训练集大小：{0}, 正样本：{1}, 负样本：{2}".format(len(train_pair), count_label(train_pair), len(train_pair)-count_label(train_pair)))
    print("验证集大小：{0}, 正样本：{1}, 负样本：{2}".format(len(valid_pair), count_label(valid_pair), len(valid_pair)-count_label(valid_pair)))
    if Augment:  # 数据增强
        # data_augment_match
        item_train_pair_augment_match_list = []
        with open(os.path.join(datafolder, 'item_train_pair_augment.json'), 'r') as file:
            item_train_pair_augment = json.load(file)
            for pair in item_train_pair_augment:
                pair = json.loads(pair)
                item_train_pair_augment_match_list.append(pair)
        # data_augment_notmatch
        item_train_pair_augment_notmatch_list = []
        match_list = []
        with open(os.path.join(datafolder, 'item_train_pair_augment.txt'), 'r') as file:
            for line in file.readlines():
                m_list = [id[1:-1] for id in line[1:-2].split(", ")]
                match_list.append(m_list)
        
        aug_num = 0
        repeat_num = 0
        while aug_num < 5300:
            repeat_num += 1
            rows = random.sample(range(len(match_list)), 2)
            row_src, row_tgt = rows
            column_src, column_tgt = random.randint(0, len(match_list[row_src]) - 1), random.randint(0, len(match_list[row_tgt]) - 1) # 不重复
            src_id = match_list[row_src][column_src]
            tgt_id = match_list[row_tgt][column_tgt]
            # 只要cate_name相同的负样本
            if id2cate[src_id] == id2cate[tgt_id]:
                temp_pair = {
                    "src_item_id": src_id,
                    "tgt_item_id": tgt_id,
                    "item_label": "0"
                }
                if temp_pair not in item_train_pair_augment_notmatch_list:
                    item_train_pair_augment_notmatch_list.append(temp_pair)
                    aug_num += 1

        print("循环次数：",repeat_num)
        random.shuffle(item_train_pair_augment_match_list)
        random.shuffle(item_train_pair_augment_notmatch_list)
        train_pair += item_train_pair_augment_match_list[:6000]
        # train_pair += item_train_pair_augment_notmatch_list
        print("数据增强后训练集大小：{0}, 正样本：{1}, 负样本：{2}".format(len(train_pair), count_label(train_pair), len(train_pair)-count_label(train_pair)))
        train_pair = remove_list_dict_duplicate(train_pair)
        print("训 练 集 去 重 后 ：{0}, 正样本：{1}, 负样本：{2}".format(len(train_pair), count_label(train_pair), len(train_pair)-count_label(train_pair)))
        random.shuffle(train_pair)

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)
    train_data = ItemDataset(item_train_info, tokenizer, max_len_main, max_len_secondary, max_len_full)
    train_load_data = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate, shuffle=True,
                                 num_workers=4, pin_memory=True)

    item_token_dict = {}
    # 生成token文件
    if not os.path.exists(os.path.join(datafolder, token_file)) or ReWrite_Token == True:
        print("~~~~~~~~~~~~~~~生成 TOKEN 文件中~~~~~~~~~~~~~~~~")
        train_item_token_list, train_item_token = [], []
        for batch in tqdm(train_load_data):
            batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full = batch
            train_item_token_list.extend({"item_id": item_id, "token_context_main": dict(token_context_main), "token_context_secondary": dict(token_context_secondary), "token_context_full": dict(token_context_full)} 
                                         for item_id, token_context_main, token_context_secondary, token_context_full 
                                         in zip(batch_item_ids, batch_token_context_main, batch_token_context_secondary, batch_token_context_full))

        with open(os.path.join(datafolder, token_file), 'w') as f:
            for item in train_item_token_list:
                item = json.dumps(item)
                train_item_token.append(item)
            json.dump(train_item_token, f, indent=4, ensure_ascii=False)
        print("~~~~~~~~~~~~~~~生成 TOKEN 文件结束~~~~~~~~~~~~~~~")

    with open(os.path.join(datafolder, token_file), 'r') as f:
        item_train_list = json.load(f)
        for item in item_train_list:
            item = json.loads(item)
            item_token_dict[item["item_id"]] = {
                "token_context_main": item["token_context_main"],
                "token_context_secondary": item["token_context_secondary"],
                "token_context_full": item["token_context_full"]
            }

    train_pair = PairDataset(train_pair, item_token_dict)
    train_load_pair = DataLoader(train_pair, batch_size=batch_size, collate_fn=train_pair.collate, num_workers=4, pin_memory=True)
    valid_pair = PairDataset(valid_pair, item_token_dict)
    valid_load_pair = DataLoader(valid_pair, batch_size=batch_size, collate_fn=train_pair.collate, num_workers=4, pin_memory=True)
    test_pair = PairDataset(test_pair, item_token_dict)
    test_load_pair = DataLoader(test_pair, batch_size=batch_size, collate_fn=train_pair.collate, num_workers=4, pin_memory=True)
    
    return train_load_pair, valid_load_pair, test_load_pair


if __name__ == '__main__':
    args = set_args_train()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    device = torch.device('cuda:0')
    max_len_main = 1
    max_len_secondary = 1
    max_len_full = 256
    epochs = args.epochs
    # 12层roberta 、batch_size 64 占用显存 11089M 、128 占用 10479 10167/ 20507
    # 24层roberta 、batch_size 128 占用显存 29177M
    batch_size = args.batch_size
    print("Epochs: ", epochs)
    print("batch_size: ", batch_size)
    learning_rate = args.learning_rate
    threshold = args.threshold
    print("threshold:", threshold)

    # datafolder = '../../data/ccks2022_data/'
    datafolder = args.datafolder
    # save_path = 'output'
    # model_path = 'output'
    save_path = args.save_path
    model_path = save_path
    Load_Model = args.Load_Model # 是否加载模型
    load_model_name = args.load_model_name
    
    # save_model_name = 'cate/baseline1_13'
    save_model_name = args.save_model_name
    
    save_model_name_auc = save_model_name + '_auc.pkl'
    save_model_name = save_model_name + '.pkl'
    # save_model_name_t = 'cate/baseline1_t.pkl'
    
    # roberta、roberta_L12、roberta_wwm、roberta_wwm_L12
    # bert_model_path = '../../bert_model/roberta_L12'
    bert_model_path = args.bert_model_path
    token_file = 'item_train_token.json'
    print("Model path :", bert_model_path)

    NoTest = args.NoTest  # True表示不要test，并入train中
    Augment = args.Augment  # 是否数据增强
    print("是否将Test并入Train：", NoTest)
    print("是否数据增强：", Augment)
    train_load_pair, valid_load_pair, test_load_pair = Get_Dataloader(NoTest, Augment, datafolder, bert_model_path, token_file, batch_size, max_len_main, max_len_secondary, max_len_full)

    encoder = BertModel.from_pretrained(bert_model_path)
    encoder_type = args.encoder_type  # "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"，"last3-avg"
    model = Baseline(encoder=encoder, output_dim=args.output_dim, encoder_type=encoder_type)
    print(model)

    # 如果有多张卡，就并行
    # 如果并行训练的话，加载已保存的模型也要并行
    # if torch.cuda.device_count() > 1:
    #     print("\nUse", torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model)

    model.to(device)
    if Load_Model:
        model.load_state_dict(torch.load(os.path.join(model_path, load_model_name)))
        print("Load model {0} success".format(load_model_name))

    bert_model_type = args.bert_model_type
    if bert_model_type == 'large':
        unfreeze_layers = ['layer.21', 'layer.22', 'layer.23', 'encoder.pooler', 'out.']
    elif bert_model_type == 'base':
        unfreeze_layers = ['layer.9', 'layer.10', 'layer.11', 'encoder.pooler', 'out.']
    
    for name, param in model.named_parameters():
        param.requires_grad = any(ele in name for ele in unfreeze_layers)
    
    #验证一下
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,param.size())

    scheduler = None
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    cosent = CoSENT(device=device)
    metrics = MetricsCalculator(beta=threshold)


    max_f1, max_auc, max_f1_t = 0.0, 0.0, 0.0
    count_epoch, count_epoch_auc, count_epoch_f1_t = 0, 0, 0
    print("encoder_type =", encoder_type)
    print('Start Training~~~')
    for epoch in range(epochs):
        train(model, train_load_pair, optimizer, scheduler, cosent, metrics, epoch)
        f1, auc, f1_t, thre_t = evaluation(model, valid_load_pair, cosent, metrics, epoch, "valid")
        # evaluation(model, test_load_pair, cosent, metrics, epoch, "test")

        # Update learning rate
        # scheduler.step()

        if max_f1 <= f1:
            if epoch > 15:
               evaluation(model, test_load_pair, cosent, metrics, epoch, "test")
            max_f1 = f1
            count_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, save_model_name))
            print("save model {0} success!".format(save_model_name))
        
        if max_auc <= auc:
            max_auc = auc
            count_epoch_auc = epoch
            # torch.save(model.state_dict(), os.path.join(save_path, save_model_name_auc))
            # print("save model {0} success!".format(save_model_name_auc))
            
        if max_f1_t <= f1_t:
            max_f1_t = f1_t
            count_epoch_f1_t = epoch


    print("\nMAX_F1 is {0:.4f}, on Epoch {1}".format(max_f1, count_epoch+1))
    print("\nMAX_F1_t is {0:.4f}, on Epoch {1}".format(max_f1_t, count_epoch_f1_t+1))
    print("\nMAX_AUC is {0:.4f}, on Epoch {1}".format(max_auc, count_epoch_auc+1))
    if not NoTest:
        print("\nStart Testing~~~")
        evaluation(model, test_load_pair, cosent, metrics, epoch, "test")


