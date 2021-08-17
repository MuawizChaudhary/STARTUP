import utils

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import itertools
from utils import to_one_hot,  AverageMeter,  loss_calc

import torch.optim as optim
import time

class Linear_Layer_Local_Loss(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_Layer_Local_Loss, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear.weight.data.zero_()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        h = self.linear(x)
        return h, h


class Final_Layer_Local_Loss(nn.Module):
    def __init__(self):
        super(Final_Layer_Local_Loss, self).__init__()

    def forward(self, x):
        return x, None


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], -1)



def optim_init(ncnn, model):
    return optim.Adam(model.parameters())
    #layer_optim = [None] * ncnn
    #for n in range(ncnn):
    #    to_train = itertools.chain(model.main_cnn.blocks[n].parameters(), model.auxillary_nets[n].parameters())
    #    layer_optim[n] = optim.Adam(to_train)
    #return layer_optim



class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature   = model_func()
        print(model_func().__class__.__name__)

        if loss_type == 'softmax':
            if model_func().__class__.__name__ =="VGGn":
                self.feature.main_cnn.blocks[-1] = Linear_Layer_Local_Loss(1024 , num_class)
            #    self.feature.main_cnn.blocks[-1] = nn.Sequential(
            #nn.AdaptiveAvgPool2d((1,1)),
            #View(512),
            #Linear_Layer_Local_Loss(512 , num_class))


            else: 
                self.feature.main_cnn.blocks[-1] = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            View(512),
            Linear_Layer_Local_Loss(512 , num_class))

        
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

        self.n_cnn = len(self.feature.main_cnn.blocks)

        self.layer_optim = optim_init(self.n_cnn, self.feature)

    def forward(self, x, n):
        scores = self.feature.forward(x)
#        scores = self.classifier.forward(out)
        return scores
    
    def train_loop(self, epoch, train_loader, optimizer, logger):
        print_freq = 150
        # avg_loss=0

        self.train()

        meters = utils.AverageMeterSet()

        end = time.time()
        for i, (x,y) in enumerate(train_loader):


            meters.update('Data_time', time.time() - end)

            representation = x.cuda(non_blocking = True)
            y = y.cuda(non_blocking = True)
            #loss=torch.tensor([0.0], requires_grad=True)
            for n in range(self.n_cnn):
                representation.detach_()
                #optimizer = self.layer_optim[n]
                pred, sim, representation = self.feature(representation, n=n)

                if n == 0:
                    loss = self.loss_fn(pred,y)
                else:
                    loss += self.loss_fn(pred, y)
                if n == self.n_cnn-1:
                    loss.backward()
                    self.layer_optim.step()
                    self.layer_optim.zero_grad()

            perf = utils.accuracy(pred.data,
                              y.data, topk=(1, 5))

            meters.update('Loss', loss.item(), 1) #return to
            meters.update('top1', perf['average'][0].item(), len(x))
            meters.update('top5', perf['average'][1].item(), len(x))

            meters.update('top1_per_class', perf['per_class_average'][0].item(), 1)
            meters.update('top5_per_class', perf['per_class_average'][1].item(), 1)

            meters.update('Batch_time', time.time() - end)
            end = time.time()

            # avg_loss = avg_loss+loss.item()
            if (i+1) % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Top1 Avg {:f} | Top5 Avg {:f}'.format(epoch, i, len(train_loader), meters.__getitem__('top1'), meters.__getitem__('top5')))
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

