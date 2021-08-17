import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import os

import models
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot, tiered_ImageNet_few_shot

from tqdm import tqdm
import pandas as pd
import argparse
import random
import copy
import warnings
from utils import to_one_hot,  AverageMeter,  loss_calc
import utils
from methods.baselinetrain import BaselineTrain




def evaluate(dataloader, params): 
    print("Loading Model: ", params.embedding_load_path)
    if params.embedding_load_path_version == 0:
        state = torch.load(params.embedding_load_path)['state']
        state_keys = list(state.keys())
        #print(state_keys)
        for _, key in enumerate(state_keys):
            if "feature." in key:
                # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
            
        sd = state
    elif params.embedding_load_path_version == 1:
        sd = torch.load(params.embedding_load_path)

        if 'epoch' in sd:
            print("Model checkpointed at epoch: ", sd['epoch'])
        sd = sd['model']
    # elif params.embedding_load_path_version == 3:
    #     state = torch.load(params.embedding_load_path)
    #     print("Model checkpointed at epoch: ", state['epoch'])
    #     state = state['model']
    #     state_keys = list(state.keys())
    #     for _, key in enumerate(state_keys):
    #         if "module." in key:
    #             # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
    #             newkey = key.replace("module.", "")
    #             state[newkey] = state.pop(key)
    #         else:
    #             state.pop(key)
    #     sd = state
    else:
        raise ValueError("Invalid load path version!")

    if params.model == 'resnet10':
        pretrained_model = models.ResNet10()
        feature_dim = pretrained_model.final_feat_dim
    elif params.model == 'resnet12':
        pretrained_model = models.Resnet12(width=1, dropout=0.1)
        feature_dim = pretrained_model.output_size
    elif params.model == 'resnet18':
        pretrained_model = models.resnet18(remove_last_relu=False, 
                                        input_high_res=True)
        feature_dim = 512
    elif params.model == 'vgg11':
        pretrained_model = models.vgg11_bn()
        pretrained_model.final_feat_dim = 512
        feature_dim = 512
    else:
        raise ValueError("Invalid model!")


    pretrained_model.load_state_dict(sd)

    model = BaselineTrain(pretrained_model, 64)
    model.load_state_dict(torch.load(params.embedding_load_path)['state'])
    pretrained_model = model


    acc_all = []

    pretrained_model.cuda()
    total = 0
    correct = 0.0

    meters = utils.AverageMeterSet()
    for i, (x, y) in tqdm(enumerate(dataloader)):
        x = x.cuda()
        y = y.cuda()

        pretrained_model.eval()
        scores = pretrained_model(x)
        
        
        perf = utils.accuracy(scores.data,
                              y.data, topk=(1, 5))
        meters.update('top1', perf['average'][0].item(), len(x))
        meters.update('top5', perf['average'][1].item(), len(x))

        #_, pred = torch.max(scores.data, 1)
        #total += y.size(0)
        #correct += (pred==y).sum().item()
       
        ###############################################################################################
   # print('Test Acc = %d %%' %
    #            (100 * correct/total))
    #print(correct, total)

    print("Top1 Avg {:f}".format(meters.__getitem__('top1')))

def main(params):

    #if params.target_dataset == 'ISIC':
    #    datamgr = ISIC_few_shot
    #elif params.target_dataset == 'EuroSAT':
    #    datamgr = EuroSAT_few_shot
    #elif params.target_dataset == 'CropDisease':
    #    datamgr = CropDisease_few_shot
    #elif params.target_dataset == 'ChestX':
    #    datamgr = Chest_few_shot
    #elif params.target_dataset == 'miniImageNet_train':
    #    datamgr = miniImageNet_few_shot
    #elif params.target_dataset == 'miniImageNet_val':
    #    datamgr = miniImageNet_few_shot
    #else:
    #    print(params.target_dataset)
    #    raise ValueError("Invalid Dataset!")

    for i in ['miniImageNet_val', 'miniImageNet_train' ]:
        params.target_dataset = i
        datamgr = miniImageNet_few_shot
    
        results = {}
        shot_done = []
        print(params.target_dataset)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(params.seed)
        torch.random.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        random.seed(params.seed)
        dataloader = datamgr.SimpleDataManager(params.image_size, params.batch_size).get_data_loader(aug=False, train_or_val=params.target_dataset=='miniImageNet_train')
        evaluate(dataloader, params)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation script')
    parser.add_argument('--target_dataset', default='miniImagenet',
                        help='test target dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Size of batch')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')
    parser.add_argument('--model', default='resnet10',
                        help='backbone architecture')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--embedding_load_path', type=str,
                        help='path to load embedding')
    parser.add_argument('--embedding_load_path_version', type=int, default=1, 
                        help='how to load the embedding')

    
    params = parser.parse_args()
    main(params)
   
