import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch import linalg as LA
import os
from torch.nn.utils.weight_norm import WeightNorm
import itertools


import math
import models
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot, tiered_ImageNet_few_shot

from tqdm import tqdm
import pandas as pd
import argparse
import random
import copy
import warnings
class auxillary_conv_classifier(nn.Module):
    def __init__(self, input_features=256, in_size=32, cnn=False,
                 num_classes=10, n_mlp=0, n_conv=0, loss_sup="pred", 
                 dim_in_decoder_arg=2048, pooling="avg",
                 bn=False, dropout=0.0):
        super(auxillary_conv_classifier, self).__init__()
        self.in_size = in_size
        self.cnn = cnn
        feature_size = input_features
        self.loss_sup = loss_sup
        input_features = in_size
        in_size = feature_size
        self.dim_in_decoder = dim_in_decoder_arg
        self.pooling = pooling
        self.pool = nn.Identity()
        self.blocks = []
        
        for n in range(n_conv):
            if bn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = nn.Identity()

            relu_temp = nn.ReLU(True)

            conv = nn.Conv2d(feature_size, feature_size,
                                 kernel_size=1, stride=1, padding=0, bias=False)

            self.blocks.append(nn.Sequential(conv, bn_temp, relu_temp))
        self.blocks = nn.ModuleList(self.blocks)

        if (loss_sup == 'pred' or loss_sup == 'predsim' or loss_sup =='zero') and pooling == "avg":
            # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = input_features, input_features
            self.dim_in_decoder = in_size * dim_out_h * dim_out_w
            while self.dim_in_decoder > dim_in_decoder_arg and ks_h < input_features:
                ks_h *= 2
                dim_out_h = math.ceil(input_features / ks_h)
                self.dim_in_decoder = in_size * dim_out_h * dim_out_w
                if self.dim_in_decoder > dim_in_decoder_arg:
                    ks_w *= 2
                    dim_out_w = math.ceil(input_features / ks_w)
                    self.dim_in_decoder = in_size * dim_out_h * dim_out_w
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - input_features // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - input_features // ks_w)) // 2
                self.pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))
                self.bn = nn.Identity()
            else:
                self.pool = nn.Identity()
                self.bn = nn.Identity()

        if pooling == "adaptiveavg":
            self.dim_in_decoder = feature_size*4
            self.pre_conv_pool = nn.AdaptiveAvgPool2d((math.ceil(self.in_size / 4), math.ceil(self.in_size / 4)))
            #self.pool = nn.AvgPool2d((2, 2))
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
        else:
            self.pre_conv_pool = nn.Identity()

        if n_mlp > 0:
            mlp_feat = self.dim_in_decoder

            layers = []

            for l in range(n_mlp):
                if bn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = nn.Identity()
                dropout_temp = torch.nn.Dropout(p=dropout, inplace=False)
                layers += [nn.Linear(mlp_feat, mlp_feat),
                           bn_temp, nn.ReLU(True), dropout_temp]

            self.mlp = True
            self.preclassifier = nn.Sequential(*layers)
            self.classifier = nn.Linear(mlp_feat, num_classes)
            self.classifier.weight.data.zero_()
            if loss_sup == 'predsim':
                self.sim_loss = nn.Conv2d(feature_size, feature_size, 3, stride=1, padding=1, bias=False)

        else:
            self.mlp = False
            self.preclassifier = nn.Identity()
            self.classifier = nn.Linear(self.dim_in_decoder, num_classes)
            self.classifier.weight.data.zero_()
            if loss_sup == 'predsim':
                self.sim_loss = nn.Conv2d(feature_size, feature_size, 3, stride=1, padding=1, bias=False)

        if not bn and not self.mlp:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(feature_size)

    def forward(self, x):
        out = None
        loss_sim = None
        if self.loss_sup == "predsim":
            loss_sim = self.sim_loss(x)

        x = self.pre_conv_pool(x)
        for block in self.blocks:
            x = block(x)
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.preclassifier(out)
        out = self.classifier(out)
        return out



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

class Classifier(nn.Module):
    def __init__(self, dim, n_way, lam_size):
        super(Classifier, self).__init__()
        self.fc = distLinear(dim, n_way) # nn.Linear(dim, n_way)#
        self.lam = nn.Parameter(torch.zeros(1, 1), requires_grad=True)

    def forward(self, x, total):
        #total.append(F.normalize(x, p=2, dim=1)*self.lam)
        total.append(x*F.sigmoid(self.lam))
        #total = torch.cat(total, dim=1)
        #total.append(F.normalize(x, p=2, dim=1)) 
        total = torch.cat(total, dim=1)
        x = self.fc(total)
        return x



class aux_class(nn.Module):
    def __init__(self, dim):
        super(aux_class, self).__init__()
        self.main = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(2),
                                    nn.Flatten())
        self.lam = nn.Parameter(torch.zeros(1, 1), requires_grad=True)
     
    def forward(self, x):
        #x = F.normalize(self.main(x), p=2, dim=1) * self.lam
        #x = F.normalize(self.main(x), p=2, dim=1) * F.sigmoid(self.lam)
        x = self.main(x) * F.sigmoid(self.lam)
        #x=F.normalize(self.main(x), p=2, dim=1)
        return x


def finetune(novel_loader, params, n_shot): 

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
        pretrained_model_template = models.ResNet10()
        feature_dim = pretrained_model_template.final_feat_dim
    elif params.model == 'resnet12':
        pretrained_model_template = models.Resnet12(width=1, dropout=0.1)
        feature_dim = pretrained_model_template.output_size
    elif params.model == 'resnet18':
        pretrained_model_template = models.resnet18(remove_last_relu=False, 
                                        input_high_res=True)
        feature_dim = 512
    elif params.model == 'vgg11':
        pretrained_model_template = models.vgg11_bn()
        feature_dim = 512
    else:
        raise ValueError("Invalid model!")

    pretrained_model_template.load_state_dict(sd)

    n_query = params.n_query
    n_way = params.n_way
    n_support = n_shot

 
    N=6

    acc_all = [[] for n in range(0,N)]
    for i, (x, y) in tqdm(enumerate(novel_loader)):

        pretrained_model = copy.deepcopy(pretrained_model_template)

        classifier = Classifier(4608, params.n_way, 128*4)
        classifier.cuda()

        pretrained_model.cuda()

        classifiers=[]
        in_plane = [[56, 64],[56, 64],[28, 128],[14, 256],[7, 512], [0, 128]]
        
        for n in range(0,N):
            l = aux_class(in_plane[n][1]*4)

            classifiers.append(l)
            #l.lam = torch.rand()
            #classifiers.append( nn.Sequential(
            #                        nn.AdaptiveAvgPool2d(2),
            #                        nn.Flatten())
            #                    )
                    #auxillary_conv_classifier(in_size=in_plane[n][0],
                    #                      input_features=in_plane[n][1],
                    #                      n_mlp=0,
                    #                      bn=False,
                    #                      pooling='adaptiveavg',
                    #                      loss_sup='pred',
                    #                      num_classes=params.n_way))
        for n in range(0,N):
            classifiers[n].cuda().train()
        ###############################################################################################
        x = x.cuda()
        x_var = x

        assert len(torch.unique(y)) == n_way
    
        batch_size = 4
        support_size = n_way * n_support 
       
        y_a_i = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()

        # split into support and query
        x_b_i = x_var[:, n_support:,: ,: ,:].contiguous().view(n_way*n_query, *x.size()[2:]).cuda() 
        x_a_i = x_var[:, :n_support,: ,: ,:].contiguous().view(n_way*n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)

        loss_fn = [nn.CrossEntropyLoss().cuda() for n in range(0,N)]
        
        to_train = list(classifier.parameters())
        to_train_1 = []
        for n in range(0,N):
            to_train_1.extend(list(classifiers[n].parameters()))

        classifier_opt = torch.optim.SGD(to_train, lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001) # experiment with this

        classifier_opt2 = torch.optim.Adadelta(to_train_1, lr=(3e+3/n_way))#SGD(to_train_1, lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001) # experiment with this
        # double check parameters are there.
        # diff optimizers for lambdas and linear layers
        # take adadelta settings and set that the default
        #classifier_opt = [torch.optim.SGD(classifiers[n].parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001) for n in range(0,N)]
                 ###############################################################################################
        

        if not params.freeze_backbone:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=0.01)

        ###############################################################################################
        total_epoch = 100
        


        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                n=-1
                total = []
                for m in pretrained_model.trunk:
                    if params.freeze_backbone:
                        m.eval()
                        with torch.no_grad():
                            if n == -1:
                                m_a_i = m(x_a_i)
                                n=0
                            else:
                                m_a_i = m(m_a_i)
                    else:
                        m.train()
        
                    #classifier_opt[n].zero_grad()
                    if not params.freeze_backbone:
                        delta_opt.zero_grad()
                        

                    #####################################
                    selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                    y_batch = y_a_i[selected_id]

                    if params.freeze_backbone:
                        output = m_a_i[selected_id]
                    else:
                        z_batch = x_a_i[selected_id]
                        output = m(z_batch)

                    if m.__class__.__name__ == "SimpleBlock" or m.__class__.__name__ == "MaxPool2d":
                        classifiers[n].train()
                        output = classifiers[n](output)
                        total.append(output)
                        #total.append(output)

                        #####################################
                        #loss.backward()

                        #classifier_opt[n].step()
                        if not params.freeze_backbone:
                            delta_opt.step()
                        n+=1

                    if m.__class__.__name__=="Flatten":
                        #total.append(F.normalize(output, p=2, dim=1))
                        classifier.train()
                        total = classifier(output, total)

                        loss = loss_fn[0](total, y_batch)
                        loss.backward()

                        classifier_opt.step()
                        classifier_opt2.step()

                        classifier_opt.zero_grad()
                        classifier_opt2.zero_grad()

        pretrained_model.eval()

        with torch.no_grad():
            n=0
            total =[]
            for m in pretrained_model.trunk:

                x_b_i = m(x_b_i)
                if m.__class__.__name__ == "SimpleBlock" or m.__class__.__name__ == "MaxPool2d":
                        classifiers[n].eval()
                        output = classifiers[n](x_b_i)
                        total.append(output)

                        n+=1




                if m.__class__.__name__=="Flatten":
                    n=0
                    classifier.eval()
                    scores = classifier(x_b_i, total)


                    y_query = np.repeat(range( n_way ), n_query )
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()
                    
                    top1_correct = np.sum(topk_ind[:,0] == y_query)
                    correct_this, count_this = float(top1_correct), len(y_query)
                    # print (correct_this/ count_this *100)
                    acc_all[n].append((correct_this/ count_this *100))

                    if (i+1) % 50 == 0:
                        acc_all_np = np.asarray(acc_all[n])
                        acc_mean = np.mean(acc_all_np)
                        acc_std = np.std(acc_all_np)
                        print('Test Acc (%d episodes) = %4.2f%% +- %4.2f%%' %
                            (len(acc_all[n]),  acc_mean, 1.96 * acc_std/np.sqrt(len(acc_all[n]))))
                                ###############################################################################################
    acc_all_array = acc_all
    for j in range(N-1):
        print(classifiers[j].lam)
        print(torch.norm(classifiers[j].lam))
    print(classifier.lam)
    print(torch.norm(classifier.lam))

    for n in range(0,1):
        acc_all  = np.asarray(acc_all_array[n])
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print("%d %d 'th Test Acc = %4.2f%% +- %4.2f%%" %
              (len(acc_all), n, acc_mean, 1.96 * acc_std/np.sqrt(len(acc_all))))

    return acc_all

def main(params):

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    if params.target_dataset == 'ISIC':
        datamgr = ISIC_few_shot
    elif params.target_dataset == 'EuroSAT':
        datamgr = EuroSAT_few_shot
    elif params.target_dataset == 'CropDisease':
        datamgr = CropDisease_few_shot
    elif params.target_dataset == 'ChestX':
        datamgr = Chest_few_shot
    elif params.target_dataset == 'miniImageNet_test':
        datamgr = miniImageNet_few_shot
    elif params.target_dataset == 'tiered_ImageNet_test':
        if params.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        datamgr = tiered_ImageNet_few_shot
    else:
        raise ValueError("Invalid Dataset!")
    
    results = {}
    shot_done = []
    print(params.target_dataset)
    for shot in params.n_shot:
        print(f"{params.n_way}-way {shot}-shot")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(params.seed)
        torch.random.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        random.seed(params.seed)
        novel_loader = datamgr.SetDataManager(params.image_size, n_eposide=params.n_episode,
                                                n_query=params.n_query, n_way=params.n_way,
                                                n_support=shot, split=params.subset_split).get_data_loader(
                                                aug=params.train_aug)
        acc_all = finetune(novel_loader, params, n_shot=shot)
        results[shot] = acc_all
        shot_done.append(shot)

        if params.save_suffix is None:
            pd.DataFrame(results).to_csv(os.path.join(params.save_dir, 
                params.source_dataset + '_' + params.target_dataset + '_' + 
                str(params.n_way) + 'way' + '.csv'), index=False)
        else:
            pd.DataFrame(results).to_csv(os.path.join(params.save_dir, 
                params.source_dataset + '_' + params.target_dataset + '_' + 
                str(params.n_way) + 'way_' + params.save_suffix + '.csv'), index=False)

    return 

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='few-shot Evaluation script')
    parser.add_argument('--save_dir', default='.', type=str, help='Directory to save the result csv')
    parser.add_argument('--source_dataset', default='miniImageNet', help='source_dataset')
    parser.add_argument('--target_dataset', default='miniImagenet',
                        help='test target dataset')
    parser.add_argument('--subset_split', type=str,
                        help='path to the csv files that contains the split of the data')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')
    parser.add_argument('--n_way', default=5, type=int,
                        help='class num to classify for training')
    parser.add_argument('--n_shot', nargs='+', default=[5], type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_episode', default=600, type=int, 
                        help='Number of episodes')
    parser.add_argument('--n_query', default=15, type=int, 
                        help='Number of query examples per class')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')
    parser.add_argument('--model', default='resnet10',
                        help='backbone architecture')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze the backbone network for finetuning')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--embedding_load_path', type=str,
                        help='path to load embedding')
    parser.add_argument('--embedding_load_path_version', type=int, default=1, 
                        help='how to load the embedding')
    parser.add_argument('--save_suffix', type=str, help='suffix added to the csv file')

    
    params = parser.parse_args()
    main(params)


class auxillary_conv_classifier(nn.Module):
    def __init__(self, input_features=256, in_size=32, cnn=False,
                 num_classes=10, n_mlp=0, n_conv=0, loss_sup="pred", 
                 dim_in_decoder_arg=2048, pooling="avg",
                 bn=False, dropout=0.0):
        super(auxillary_conv_classifier, self).__init__()
        self.in_size = in_size
        self.cnn = cnn
        feature_size = input_features
        self.loss_sup = loss_sup
        input_features = in_size
        in_size = feature_size
        self.dim_in_decoder = dim_in_decoder_arg
        self.pooling = pooling
        self.pool = nn.Identity()
        self.blocks = []
        
        for n in range(n_conv):
            if bn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = nn.Identity()

            relu_temp = nn.ReLU(True)

            conv = nn.Conv2d(feature_size, feature_size,
                                 kernel_size=1, stride=1, padding=0, bias=False)

            self.blocks.append(nn.Sequential(conv, bn_temp, relu_temp))
        self.blocks = nn.ModuleList(self.blocks)

        if (loss_sup == 'pred' or loss_sup == 'predsim' or loss_sup =='zero') and pooling == "avg":
            # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = input_features, input_features
            self.dim_in_decoder = in_size * dim_out_h * dim_out_w
            while self.dim_in_decoder > dim_in_decoder_arg and ks_h < input_features:
                ks_h *= 2
                dim_out_h = math.ceil(input_features / ks_h)
                self.dim_in_decoder = in_size * dim_out_h * dim_out_w
                if self.dim_in_decoder > dim_in_decoder_arg:
                    ks_w *= 2
                    dim_out_w = math.ceil(input_features / ks_w)
                    self.dim_in_decoder = in_size * dim_out_h * dim_out_w
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - input_features // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - input_features // ks_w)) // 2
                self.pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))
                self.bn = nn.Identity()
            else:
                self.pool = nn.Identity()
                self.bn = nn.Identity()

        if pooling == "adaptiveavg":
            self.dim_in_decoder = feature_size*4
            self.pre_conv_pool = nn.AdaptiveAvgPool2d((math.ceil(self.in_size / 4), math.ceil(self.in_size / 4)))
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
        else:
            self.pre_conv_pool = nn.Identity()

        if n_mlp > 0:
            mlp_feat = self.dim_in_decoder

            layers = []

            for l in range(n_mlp):
                if bn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = nn.Identity()
                dropout_temp = torch.nn.Dropout(p=dropout, inplace=False)
                layers += [nn.Linear(mlp_feat, mlp_feat),
                           bn_temp, nn.ReLU(True), dropout_temp]

            self.mlp = True
            self.preclassifier = nn.Sequential(*layers)
            self.classifier = nn.Linear(mlp_feat, num_classes)
            self.classifier.weight.data.zero_()
            if loss_sup == 'predsim':
                self.sim_loss = nn.Conv2d(feature_size, feature_size, 3, stride=1, padding=1, bias=False)

        else:
            self.mlp = False
            self.preclassifier = nn.Identity()
            self.classifier = nn.Linear(self.dim_in_decoder, num_classes)
            self.classifier.weight.data.zero_()
            if loss_sup == 'predsim':
                self.sim_loss = nn.Conv2d(feature_size, feature_size, 3, stride=1, padding=1, bias=False)

        if not bn and not self.mlp:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(feature_size)

    def forward(self, x):
        out = None
        loss_sim = None
        if self.loss_sup == "predsim":
            loss_sim = self.sim_loss(x)

        x = self.pre_conv_pool(x)
        for block in self.blocks:
            x = block(x)
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.preclassifier(out)
        out = self.classifier(out)
        return out, loss_sim


