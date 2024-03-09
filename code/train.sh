# 用 roberta_base 模型生成 128 维文本emb
python ./text/baseline1.py --datafolder ../data/text/ --save_path ./output --save_model_name baseline1_text_base_1 --bert_model_type base --bert_model_path ../pretrained_model/roberta_L12 --cuda_visible_devices 0

# 用 roberta_large 模型生成 128 维文本emb
python ./text/baseline1.py --datafolder ../data/text/ --save_path ./output --save_model_name baseline1_text_large_1 --bert_model_type large --bert_model_path ../pretrained_model/roberta --cuda_visible_devices 0

# 用 swin_transformer 生成 512 维图片emb
python ./image/image_swin_final_512.py --CUDA_VISIBLE_DEVICES 0 --datafolder ../data/pic --save_path output/image_model --pretrain_image_model_path ../pretrained_model/swin_large_patch4_window12_384_in22k

# 用 swin_transformer 生成 256 维图片emb
python ./image/image_swin_final_256.py --CUDA_VISIBLE_DEVICES 0 --datafolder ../data/pic --save_path output/image_model --pretrain_image_model_path ../pretrained_model/swin_large_patch4_window12_384_in22k