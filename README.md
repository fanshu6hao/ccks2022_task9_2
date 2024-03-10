# ccks2022_task9_2
CCKS2022-任务九-子任务二-**基于知识图谱的商品同款挖掘**-第一名 USTC-MINE 方案

[比赛链接](https://tianchi.aliyun.com/competition/entrance/531956)

[论文链接](https://link.springer.com/chapter/10.1007/978-981-19-8300-9_20)
  
    @inproceedings{ccks22_task9_2,
      author       = {Changkai Feng and
                      Wei Chen and
                      Chao Chen and
                      Tong Xu and
                      Enhong Chen},
      title        = {Multimodal Representation Learning-Based Product Matching},
      booktitle    = {{CCKS} 2022 - Evaluation Track - 7th China Conference on Knowledge
                      Graph and Semantic Computing Evaluations},
      series       = {Communications in Computer and Information Science},
      volume       = {1711},
      pages        = {180--190},
      year         = {2022},
      url          = {https://doi.org/10.1007/978-981-19-8300-9\_20},
      doi          = {10.1007/978-981-19-8300-9\_20}
    }

# 目录结构
    code  
      
      ---code/output       训练过程中保存的模型，预测产生的csv和jsonl文件
      
      ---code/text         文本模型代码
      
      ---code/image        图片模型代码
      
      ---submit_single.py  用单个csv生成提交的jsonl文件
      
      ---concat_emb.py     concat两个csv embedding
      
      ---submit_rules.py   将最后的文本和图片emb csv文件加上规则生成jsonl提交文件
    
    data  
    
      ---data/pic          图片数据，包含 item_train_images 和 item_test_images 两个文件夹
      
      ---data/text         文本数据，下载原始数据后需要预处理
    
    model             
    
    pretrained_model

---
# 运行环境说明
     GPU型号：Tesla V100-SXM2-32GB
     
     cuda版本：10.2
     
     Python版本：3.6.13
     
     pytorch版本：1.7.1
     
     train.sh本地运行大致时长：50h (图像模型) + 10h（文本模型roberta_base）+ 20h（文本模型large）
     
     predict.sh本地运行大致时长：1h (图像预测) + 20min（文本模型）

---
# 注意事项
     1. 由于 image_swin_final_512.py 以及 image_swin_final_256.py 图像训练过程中将每个epoch模型都保存下来，故使用 predict.sh 预测时需修改 --image_model_name 参数来选择训练数据所划分的 valid 和 test F1 值都较大的模型。
     2. 本任务中阈值动态变化，由于训练过程的误差以及模型融合结果的不确定性，最后提交文件的阈值也无法确定，我们在复赛中得分最高的提交文件中所选择的阈值为0.776（加上规则后），根据的原则是选择这一阈值将得到4900个左右的正样本，故可按照该原则动态选择阈值。
     3. 代码中没有使用外部数据，我们对文本数据 (item_train_info.jsonl) 进行了预处理，对图像数据没有任何修改。
     4. 文本模型 roberta_base 本地 valid F1 约 0.862，roberta_large 约 0.864；图片模型 swin_transformer 本地 valid F1 约0.902

---
# 运行流程
     1. pip install -r requirements.txt 创建环境
     2. 执行 train.sh 训练文本和图像模型(共4个)
     3. 执行 predict.sh ensemble 生成融合后的 embedding
        
    (可选项：单独生成文本、图像embedding)
    
    predict.sh model_text_base
    
    predict.sh model_text_large
    
    predict.sh model_image_256
    
    predict.sh model_image_512
