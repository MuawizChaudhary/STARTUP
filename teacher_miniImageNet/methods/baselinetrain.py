import utils

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

import time



class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores


class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature   = model_func()

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = distLinear(self.feature.final_feat_dim, num_class)
        
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        print(self.feature.final_feat_dim, num_class)

    def forward(self, x):
        x = x.cuda()
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        y = y.cuda()

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer, logger):
        print_freq = 150
        # avg_loss=0

        self.train()

        meters = utils.AverageMeterSet()

        end = time.time()
        for i, (X,y) in enumerate(train_loader):
            meters.update('Data_time', time.time() - end)

            optimizer.zero_grad()
            logits = self.forward(X)

            y = y.cuda()
           # print(torch.min(y))
            print(y)
            loss = self.loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            perf = utils.accuracy(logits.data,
                              y.data, topk=(1, 5))

            meters.update('Loss', loss.item(), 1)
            meters.update('top1', perf['average'][0].item(), len(X))
            meters.update('top5', perf['average'][1].item(), len(X))

            meters.update('top1_per_class', perf['per_class_average'][0].item(), 1)
            meters.update('top5_per_class', perf['per_class_average'][1].item(), 1)

            meters.update('Batch_time', time.time() - end)
            end = time.time()

            # avg_loss = avg_loss+loss.item()
            if (i+1) % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
               # print('Epoch {:d} | Batch {:d}/{:d} | Top1 Avg {:f} | Top5 Avg {:f} '.format(epoch, i, len(train_loader), meters.__getitem__('top1'), meters.__getitem__('top5')))
                logger_string = ('Training Epoch: [{epoch}] Step: [{step} / {steps}] Batch Time: {meters[Batch_time]:.4f} '
                             'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                             'Top1: {meters[top1]:.4f} Top5: {meters[top5]:.4f} '
                             'Top1_per_class: {meters[top1_per_class]:.4f} '
                             'Top5_per_class: {meters[top5_per_class]:.4f} ').format(
                    epoch=epoch, step=i+1, steps=len(train_loader), meters=meters)

                logger.info(logger_string)
        
        logger_string = ('Training Epoch: [{epoch}] Step: [{step}] Batch Time: {meters[Batch_time]:.4f} '
                     'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                     'Top1: {meters[top1]:.4f} Top5: {meters[top5]:.4f} '
                     'Top1_per_class: {meters[top1_per_class]:.4f} '
                     'Top5_per_class: {meters[top5_per_class]:.4f} ').format(
                    epoch=epoch+1, step=0, meters=meters)

        logger.info(logger_string)

        return meters.averages()

            
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration

