"""Contains losses"""

import torch
from torch.nn import functional as F
import numpy as np


def dice_loss(output, target):
    """
    input is a torch variable of size BatchxCxHxWxD representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    if len(target.size()) == 5:
        target = target.squeeze(1)
    
    # probs = B x C x H x W x D
    probs = F.softmax(output, dim=1)
        
    # target = B x H x W x D    
    target = F.one_hot(target.long(), num_classes=probs.size()[1])
    # target = B x C x H x W x D
    target = target.permute(0, 4, 1, 2, 3)
    
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total