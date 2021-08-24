""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import random

__all__ = ['resnet152', 'resnet18', 'resnet34', 'resnet50',  'resnet101']

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


class auxillary_linear_classifier(nn.Module):
    def __init__(self, input_features=256,
                 num_classes=10, n_mlp=0, 
                 loss_sup="pred", bn=False, dropout=0.0):
        super(auxillary_linear_classifier, self).__init__()
        feature_size = input_features
        self.loss_sup = loss_sup
        #dropout =0.0

        if n_mlp > 0:
            self.mlp = True
            mlp_feat = feature_size
            layers = []

            for l in range(n_mlp):
                in_feat = mlp_feat
                if bn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = nn.Identity()
                dropout_temp = torch.nn.Dropout(p=dropout, inplace=False)
                layers += [nn.Linear(in_feat, mlp_feat),
                           bn_temp, nn.ReLU(True), dropout_temp]

            self.preclassifier = nn.Sequential(*layers)
            self.classifier = nn.Linear(mlp_feat, num_classes)
            self.classifier.weight.data.zero_()

            if loss_sup == 'predsim':
                self.sim_loss = nn.Linear(feature_size, feature_size, bias=False)
        else:
            self.mlp = False
            self.preclassifier = nn.Identity()
            self.classifier = nn.Linear(feature_size, num_classes)
            self.classifier.weight.data.zero_()

            if loss_sup == 'predsim':
                self.sim_loss = nn.Linear(feature_size, feature_size, bias=False)

        if not bn or not self.mlp:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm1d(feature_size)

    def forward(self, x):
        out = None
        loss_sim = None

        if self.loss_sup == "predsim":
            loss_sim = self.sim_loss(x)

        out = self.preclassifier(x)
        out = self.classifier(out)
        return out, loss_sim

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], -1)


class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks
    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n

        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x
        out = self.blocks[n](x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    expansion = 1


    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
    
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)

        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


def ResNet10_dgl():
    return ResNet(SimpleBlock, [1,1,1,1])

class Classifier(nn.Module):
  def __init__(self, channel_dim, feature_dim, num_classes, groupings):
    super(Classifier, self).__init__()
    self.length = len(groupings)
    self.channel_length = channel_dim//self.length
    self.register_buffer('groupings', torch.tensor(groupings))
    print(groupings)
    self.linears = nn.ModuleList([nn.Linear(feature_dim*self.channel_length, num_classes) for _ in groupings])
    self.flatten = nn.Flatten()

  def forward(self, x):
    x_p = []

    # anyway to parallelize this computation?
    for i in range(self.length):
      selected_channels = self.flatten(x[:,i * self.channel_length: (i+1) * self.channel_length])
      x_p.append(self.linears[i](selected_channels))
    
    return x_p

#class Classifier(nn.Module):
#  def __init__(self, channel_dim, feature_dim, num_classes, num_blocks):
#    super(Classifier, self).__init__()
#    self.aux_classifier = nn.Sequential( 
#            #nn.AdaptiveAvgPool2d((2)),
#            nn.Flatten(),
#            nn.Linear(channel_dim*feature_dim, num_classes)
#            )
#
#  def forward(self, x):
#    return self.aux_classifier(x)
#
#

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=64, split_points=0, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.avg_size = int(56 / 2)
        self.in_size = 56
        self.counter = 0
        blocks = nn.ModuleList([])
        base_blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])
        self.auxillary_size_tracker = []

        n_mlp = 0
        bn = False

        groupings = [i for i in range(0, num_classes)]
        random.shuffle(groupings)
        groupings = [[groupings[i] for i in range(0, num_classes//2)], [groupings[i] for i in range(num_classes//2, num_classes)]]
        print(groupings)
        self.groupings = groupings

        ## Initial layer
        layer = [nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3,bias=False),
                 nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        base_blocks.append(nn.Sequential(*layer))
        self.auxillary_size_tracker.append((self.in_size,self.inplanes))

        base_blocks = self._make_layer(block, base_blocks, 64, layers[0], **kwargs)
        base_blocks = self._make_layer(block, base_blocks, 128, layers[1], stride=2, **kwargs)
        base_blocks = self._make_layer(block, base_blocks, 256, layers[2], stride=2, **kwargs)
        base_blocks = self._make_layer(block, base_blocks, 512, layers[3], stride=2, **kwargs)

        self.final_feat_dim=512
 
        if split_points != 0:

            len_layers = len(base_blocks)
            split_depth = math.ceil(len(base_blocks) / split_points)

            for splits_id in range(split_points):
                left_idx = splits_id * split_depth
                right_idx = (splits_id + 1) * split_depth
                if right_idx > len_layers:
                    right_idx = len_layers
                blocks.append(nn.Sequential(*base_blocks[left_idx:right_idx]))
                in_size, planes = self.auxillary_size_tracker[right_idx-1]
                self.auxillary_nets.append(
                    auxillary_conv_classifier(in_size=in_size,
                                          input_features=planes,
                                          n_mlp=n_mlp,
                                          bn=bn,
                                          pooling='adaptiveavg',
                                          loss_sup='pred',
                                          num_classes=num_classes)
                )

            blocks.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            View(512 * block.expansion),
            Linear_Layer_Local_Loss(512 * block.expansion, num_classes))
            )
            self.auxillary_nets.append(nn.Identity())


        else:
            for i in range(len(base_blocks)):
                blocks.append(base_blocks[i])
                in_size, planes = self.auxillary_size_tracker[i] 
                print(in_size, planes, num_classes)
                self.auxillary_nets.append(Classifier(planes, in_size*in_size, num_classes,len(base_blocks))) #len(base_blocks)))


                #self.auxillary_nets.append(
                #    auxillary_conv_classifier(in_size=in_size,
                #                          input_features=planes,
                #                          n_mlp=n_mlp,
                #                          bn=bn,
                #                          pooling='adaptiveavg',
                #                          loss_sup='pred',
                #                          num_classes=num_classes)
                #)
#Classifier(nn.Module):
#  def __init__(self, planes, in_size*in_size, num_classes, groupings):

            blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                View(512 * block.expansion),
                Linear_Layer_Local_Loss(512 * block.expansion, num_classes))
                )
            self.auxillary_nets.append(nn.Identity())#(Final_Layer_Local_Loss())


        self.main_cnn = rep(blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, base_blocks, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            self.avg_size = int(self.avg_size/2)
            self.in_size =  int(self.in_size/2)
        if self.counter == 0:
            self.counter = 1
            half_res=False
        else:
            half_res=True

        base_blocks.append(block(self.inplanes, planes,half_res))
        self.auxillary_size_tracker.append((self.in_size,planes*block.expansion))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            base_blocks.append(block(self.inplanes, planes, half_res))
            self.auxillary_size_tracker.append((self.in_size,planes*block.expansion))
        return base_blocks


    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)

        pred = self.auxillary_nets[n](representation)
        return pred, representation

class Linear_Layer_Local_Loss(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_Layer_Local_Loss, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear.weight.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        return h, h


class Final_Layer_Local_Loss(nn.Module):
    def __init__(self):
        super(Final_Layer_Local_Loss, self).__init__()

    def forward(self, x):
        return x, None



def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
