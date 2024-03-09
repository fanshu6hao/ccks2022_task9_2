import argparse


# 训练
def set_args_train():
    parser = argparse.ArgumentParser('Training')
    
    parser.add_argument('--epochs', default=30, type=int, help='epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning_rate')
    parser.add_argument('--threshold', default=0.8, type=float, help='阈值')
    parser.add_argument('--NoTest', default=False, type=bool, help='是否将Test数据加入训练')
    parser.add_argument('--Augment', default=False, type=bool, help='是否数据增强')
    parser.add_argument('--datafolder', default='../../data/ccks2022_data/', type=str, help='训练数据路径')
    parser.add_argument('--save_path', default='output', type=str, help='保存的模型路径')
    parser.add_argument('--save_model_name', default='model_text_base1', type=str, help='保存的模型名称')
    parser.add_argument('--bert_model_type', default='base', type=str, help='base或large, 用于代码中选择冻结的层')
    parser.add_argument('--bert_model_path', default='../../bert_model/roberta_L12', type=str, help='预训练模型路径')
    parser.add_argument('--Load_Model', default=False, type=bool, help='是否加载模型继续训练')
    parser.add_argument('--load_model_name', default=None, type=str, help='加载继续训练的模型名称')
    parser.add_argument('--encoder_type', default="last-avg", type=str, help='bert模型输出方式')  # "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"，"last3-avg"
    parser.add_argument('--output_dim', default=128, type=int, help='最后生成的embedding维度')
    parser.add_argument('--cuda_visible_devices', default='0', type=str, help='可见GPU')
    
    return parser.parse_args()


# 预测
def set_args_predict():
    parser = argparse.ArgumentParser('Predicting')
    
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--threshold', default=0.8, type=float, help='阈值')
    parser.add_argument('--datafolder', default='../../data/ccks2022_data/', type=str, help='test数据路径')
    parser.add_argument('--save_path', default='output', type=str, help='保存的csv文件路径')
    parser.add_argument('--model_path', default='output', type=str, help='加载的模型路径')
    parser.add_argument('--load_model_name', default='cate/baseline1_12', type=str, help='加载的模型名称')
    parser.add_argument('--save_csv_name', default='pair_fea_list_1', type=str, help='保存的csv文件名称')
    parser.add_argument('--bert_model_path', default='../../bert_model/roberta_L12', type=str, help='预训练模型路径')
    parser.add_argument('--encoder_type', default="last-avg", type=str, help='bert模型输出方式')  # "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"，"last3-avg"
    parser.add_argument('--output_dim', default=128, type=int, help='最后生成的embedding维度')
    parser.add_argument('--cuda_visible_devices', default='0', type=str, help='可见GPU')
    
    return parser.parse_args()



# concat 两个embedding csv文件成一个
def set_args_concat():
    parser = argparse.ArgumentParser('Concating')
    
    parser.add_argument('--csv1', type=str, default='', help='生成的embedding csv文件1')
    parser.add_argument('--csv2', type=str, default='', help='生成的embedding csv文件2')
    parser.add_argument('--save_csv', type=str, default='', help='合并后的csv文件')
    parser.add_argument('--test_pair', type=str, default='', help='item_test_pair.jsonl文件')
    
    return parser.parse_args()


# 生成提交的jsonl文件，加上规则
def set_args_submit():
    parser = argparse.ArgumentParser('Submitting')
    
    parser.add_argument('--csv_text', type=str, default='', help='文本的embedding csv文件')
    parser.add_argument('--csv_pic', type=str, default='', help='图片的embedding csv文件')
    parser.add_argument('--test_info', type=str, default='', help='item_test_info.csv文件')
    parser.add_argument('--test_pair', type=str, default='', help='item_test_pair.jsonl文件')
    parser.add_argument('--save_file_name', type=str, default='', help='最终的jsonl结果文件')
    parser.add_argument('--add_rules', type=bool, default=True, help='是否加上规则')
    parser.add_argument('--threshold', type=float, default=0.776, help='阈值')
    
    return parser.parse_args()


# 单个csv生成提交的jsonl文件，
def set_args_submit_single():
    parser = argparse.ArgumentParser('Submit_single')
    
    parser.add_argument('--csv_file', type=str, default='', help='文本的embedding csv文件')
    parser.add_argument('--test_info', type=str, default='', help='item_test_info.csv文件')
    parser.add_argument('--test_pair', type=str, default='', help='item_test_pair.jsonl文件')
    parser.add_argument('--save_file_name', type=str, default='', help='最终的jsonl结果文件')
    parser.add_argument('--threshold', type=float, default=0.8, help='阈值')
    
    return parser.parse_args()