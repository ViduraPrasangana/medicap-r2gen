python main.py \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 16 \
--epochs 30 \
--save_dir results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--seed 456789 ${@:1}
 