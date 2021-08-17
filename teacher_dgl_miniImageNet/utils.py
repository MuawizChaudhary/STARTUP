import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_one_hot = F.one_hot(y, num_classes=n_dims).float()
    return y_one_hot


def similarity_matrix(x, no_similarity_std):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R


def loss_calc(y_hat_local, Rh, y, y_onehot, args):#loss_sup, beta, no_similarity_std):
    if args.loss_sup == 'pred':
        loss_sup = F.cross_entropy(y_hat_local,  y)
    elif args.loss_sup == 'predsim':
        if Rh is not None:
            Rh = similarity_matrix(Rh, args.no_similarity_std)
            Ry = similarity_matrix(y_onehot, args.no_similarity_std).detach()
            loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local, y)
            loss_sim = args.beta * F.mse_loss(Rh, Ry)
            loss_sup = loss_pred + loss_sim
        else:
            loss_sup = (1-args.beta) * F.cross_entropy(y_hat_local,  y)
    elif args.loss_sup == 'zero':
        loss_sup = F.cross_entropy(y_hat_local,  y)
        loss_sup = torch.zeros_like(loss_sup)
    return loss_sup


# Test routines

#### Some helper functions
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
            
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
                        
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(epoch, model, test_loader, cuda=True, num_classes=10):
    ''' Run model on test set '''
    model.eval()
    correct = 0

    # Loop test set
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        h, y,  = data, target

        for n in range(len(model.main_cnn.blocks)):
            output, sim, h = model(h, n=n)

        pred = output.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()

    error_percent = 100.0 * float(correct) / len(test_loader.dataset)
    print('acc: ' + str(error_percent))

    return error_percent


def validate(val_loader, model, epoch, n, loss_sup, iscuda):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        total = 0
        for i, (input, target) in enumerate(val_loader):
            if iscuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

            representation = input
            for i in range(n):
                output, _, representation = model(representation, n=i)
            output, _, representation = model(representation, n=n)
            
            # measure accuracy and record loss
            loss = F.cross_entropy(output, target)
            prec1 = accuracy(output.data, target)
            losses.update(float(loss.item()), float(input.size(0)))
            top1.update(float(prec1[0]), float(input.size(0)))

            total += input.size(0)
        wandb.log({"Layer " + str(n) + " test loss": losses.avg}, step=epoch)
        wandb.log({"Layer " + str(n) + " top1": top1.avg}, step=epoch)
    return top1.avg
