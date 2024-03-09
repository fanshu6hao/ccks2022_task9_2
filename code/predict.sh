echo $0
if [[ ${1} = "ensemble" ]]
then
    # 文本base
    # 这里的save_path是指 保存的csv的路径
	python ./text/predict.py --datafolder ../data/text/ --save_path ./output --save_csv_name pair_fea_list_text_base1 --bert_model_path ../pretrained_model/roberta_L12 --model_path ../model --load_model_name baseline1_text_base --cuda_visible_devices 0
    # 文本large
    python ./text/predict.py --datafolder ../data/text/ --save_path ./output --save_csv_name pair_fea_list_text_large1 --bert_model_path ../pretrained_model/roberta --model_path ../model --load_model_name baseline1_text_large --cuda_visible_devices 0
    # 图片512
    # 这里的save_path是指 模型的路径
    python image/predict_image_512.py --CUDA_VISIBLE_DEVICES 0 --datafolder ../data/text --image_path ../data/pic/item_test_images --save_path ../model --image_model_name baseline_image_large_512.pkl --save_csv_name output/pair_fea_list_image_512.csv --pretrain_image_model_path ../pretrained_model/swin_large_patch4_window12_384_in22k
    # 图片256
    python image/predict_image_256.py --CUDA_VISIBLE_DEVICES 0 --datafolder ../data/text --image_path ../data/pic/item_test_images --save_path ../model --image_model_name baseline_image_large_256.pkl --save_csv_name output/pair_fea_list_image_256.csv --pretrain_image_model_path ../pretrained_model/swin_large_patch4_window12_384_in22k

    # 合并两个文本emb
    python concat_emb.py --csv1 ./output/pair_fea_list_text_base1.csv --csv2 ./output/pair_fea_list_text_large1.csv --save_csv ./output/pair_fea_list_text.csv --test_pair ../data/text/item_test_pair.jsonl

    # 合并两个图片emb
    python concat_emb.py --csv1 ./output/pair_fea_list_image_512.csv --csv2 ./output/pair_fea_list_image_256.csv --save_csv ./output/pair_fea_list_image.csv --test_pair ../data/text/item_test_pair.jsonl

    # 生成最终的提交文件
    # 若去掉规则，则将 add_rules 改为 False，threshold也可以适当修改
    python submit_rules.py --add_rules True --threshold 0.776 --save_file_name ./output/fanshu_result.jsonl --csv_text ./output/pair_fea_list_text.csv --csv_pic ./output/pair_fea_list_image.csv --test_info ../data/text/item_test_info.csv --test_pair ../data/text/item_test_pair.jsonl
elif [[ ${1} = "model_text_base" ]]
then
	python ./text/predict.py --datafolder ../data/text/ --save_path ./output --save_csv_name pair_fea_list_text_base1 --bert_model_path ../pretrained_model/roberta_L12 --model_path ../model --load_model_name baseline1_text_base --cuda_visible_devices 0
    # 阈值可以适当修改
    python submit_single.py --threshold 0.8 --save_file_name ./output/text_base_result.jsonl --csv_file ./output/pair_fea_list_text_base1.csv --test_info ../data/text/item_test_info.csv --test_pair ../data/text/item_test_pair.jsonl
elif [[ ${1} = "model_text_large" ]]
then
	python ./text/predict.py --datafolder ../data/text/ --save_path ./output --save_csv_name pair_fea_list_text_large1 --bert_model_path ../pretrained_model/roberta --model_path ../model --load_model_name baseline1_text_large --cuda_visible_devices 0
    # 阈值可以适当修改
    python submit_single.py --threshold 0.8 --save_file_name ./output/text_large_result.jsonl --csv_file ./output/pair_fea_list_text_large1.csv --test_info ../data/text/item_test_info.csv --test_pair ../data/text/item_test_pair.jsonl
elif [[ ${1} = "model_image_256" ]]
then
	python image/predict_image_256.py --CUDA_VISIBLE_DEVICES 0 --datafolder ../data/text --image_path ../data/pic/item_test_images --save_path ../model --image_model_name baseline_image_large_256.pkl --save_csv_name output/pair_fea_list_image_256.csv --pretrain_image_model_path ../pretrained_model/swin_large_patch4_window12_384_in22k
    # 阈值可以适当修改
    python submit_single.py --threshold 0.8 --save_file_name ./output/image_256_result.jsonl --csv_file ./output/pair_fea_list_image_256.csv --test_info ../data/text/item_test_info.csv --test_pair ../data/text/item_test_pair.jsonl
elif [[ ${1} = "model_image_512" ]]
then
	python image/predict_image_512.py --CUDA_VISIBLE_DEVICES 0 --datafolder ../data/text --image_path ../data/pic/item_test_images --save_path ../model --image_model_name baseline_image_large_512.pkl --save_csv_name output/pair_fea_list_image_512.csv --pretrain_image_model_path ../pretrained_model/swin_large_patch4_window12_384_in22k
    # 阈值可以适当修改
    python submit_single.py --threshold 0.8 --save_file_name ./output/image_512_result.jsonl --csv_file ./output/pair_fea_list_image_512.csv --test_info ../data/text/item_test_info.csv --test_pair ../data/text/item_test_pair.jsonl
fi