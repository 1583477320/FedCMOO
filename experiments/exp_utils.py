import urllib
import torch
import torch.nn.functional as F 


# Adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/losses.py
def nll(pred, gt, val=False):
    if val:
        return F.nll_loss(pred, gt, size_average=False)
    else:
        return F.nll_loss(pred, gt)

def rmse(pred, gt, val=False):
    pass

def cross_entropy2d(input, target, weight=None, val=False):
    if val:
        size_average = False
    else:
        size_average = True 
    
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def l1_loss_depth(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target > 0
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss 


def l1_loss_instance(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target!=250
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss 