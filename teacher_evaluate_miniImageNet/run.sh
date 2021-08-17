#!/bin/bash

#jeval "$(conda shell.bash hook)"
#jconda activate venvfewshot
export CUDA_VISIBLE_DEVICES=7

# training a classification model on miniImageNet
python train.py --dataset miniImageNet --model VGG11  --method baseline --bsize 128 --start_epoch 0 --stop_epoch 400 --train_aug 

# training a classification model on tieredImageNet
# python train.py --dataset tiered_ImageNet --model ResNet12  --method baseline --bsize 256 --start_epoch 0 --stop_epoch 90
