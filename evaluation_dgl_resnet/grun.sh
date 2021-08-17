#!/bin/bash

# bash script to evaluate different representations. 
# finetune.py learns a linear classifier on the features extracted from the support set 
# compile_result.py computes the averages and the 96 confidence intervals from the results generated from finetune.py
# and evaluate on the query set
export CUDA_VISIBLE_DEVICES=5
for source in "miniImageNet"
do
    for target in  "EuroSAT" 
    do
        # TODO: Please set the following argument appropriately "ChestX" "ChestX"  "miniImageNet_test" 
        # --save_dir: directory to save the results from evaluation
        # --embedding_load_path: representation to be evaluated 
        # --embedding_load_path_version: either 0 or 1. This is 1 most of the times. Only set this to 0 when 
        #                                evaluating teacher model trained using teacher_miniImageNet/train.py
        # E.g. the following command evaluates the STARTUP representation on 600 tasks
        #      and save the results of the 600 tasks at results/STARTUP_miniImageNet/$source\_$target\_5way.csv
        python eval.py \
        --image_size 224 \
        --n_way 5 \
        --n_shot 1 5 20 \
        --n_episode 100 \
        --n_query 15 \
        --seed 1 \
	--freeze_backbone \
        --save_dir results/STARTUP_miniImageNet \
        --source_dataset $source \
        --target_dataset $target \
        --subset_split datasets/split_seed_1/$target\_labeled_80.csv \
        --model resnet10_dgl \
        --embedding_load_path   /local/oyallon/muawiz/models_mlp_1_bn/checkpoints/miniImageNet/ResNet10_dgl_baseline_256_aug/399.tar\
        --embedding_load_path_version 0
    done
done
# _split_2
