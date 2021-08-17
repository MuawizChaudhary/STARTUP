#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# training a classification model on miniImageNet
python train_avg.py --dataset miniImageNet --model ResNet10_avg  --method baseline --bsize 256 --start_epoch 0 --stop_epoch 400 --train_aug 

# training a classification model on tieredImageNet
# python train.py --dataset tiered_ImageNet --model ResNet10_dgl  --method baseline --bsize 256 --start_epoch 0 --stop_epoch 90
