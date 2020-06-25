"""Contains losses"""

import torch
from torch.nn import functional as F
import numpy as np


def dice_loss(output, target):
    
    if len(target.size()) == 5:
        target = target.squeeze(1) 
    
    target = F.one_hot(target, num_classes=2)
    target = target.permute(0, 4, 1, 2, 3)
    
    pred = F.softmax(output, dim=1)
    
    dim = tuple(range(2, len(pred.size())))
    
    num = pred * target         # b,c,h,w,d
    num = torch.sum(num, dim=dim) # b, c
    
    den1 = target**2
    den1 = torch.sum(den1, dim=dim)

    den2 = pred**2
    den2 = torch.sum(den2, dim=dim) 
    
    dice = 2*(num + 0.001)/(den1 + den2 + 0.001)
    dice_eso = dice[:,1:]       # ignore background dice val, and take the foreground

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0) # divide by batch_sz
    
    return dice_total
