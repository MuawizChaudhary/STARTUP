#!/bin/bash

# bash script to evaluate different representations. 
# finetune.py learns a linear classifier on the features extracted from the support set 
# compile_result.py computes the averages and the 96 confidence intervals from the results generated from finetune.py
# and evaluate on the query set
export CUDA_VISIBLE_DEVICES=6

for source in "miniImageNet"
do
    for target in "miniImageNet_train" 
    do
        python finetune.py \
        --image_size 224 \
        --batch_size 64 \
        --seed 1 \
        --target_dataset $target \
        --model resnet10 \
	--embedding_load_path /local/oyallon/muawiz/models_dist/checkpoints/miniImageNet/ResNet10_baseline_256_aug/399.tar\
        --embedding_load_path_version 0
    done
done
