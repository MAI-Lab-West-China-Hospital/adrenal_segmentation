# -*- coding: utf-8 -*-

from monai.losses import FocalLoss, DiceLoss
import torch
import torch.nn as nn

CEloss = nn.CrossEntropyLoss()

dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

focal_loss = FocalLoss(gamma=2.0)

class DiceCELoss(nn.Module):
    def __init__(self, w=0.1):
        super(DiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.w = w

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss1 = self.dice_loss(input, target)
        loss2 = self.ce_loss(input, target.squeeze().long())
        loss = loss1 + self.w * loss2
        return loss


class DiceFocalLoss(nn.Module):
    def __init__(self, w=0.5):
        super(DiceFocalLoss, self).__init__()
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.w = w

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss1 = self.dice_loss(input, target)
        loss2 = self.focal_loss(input, target)
        loss = loss1 + self.w * loss2
        return loss


class MAELoss(nn.MSELoss):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predict, soft_y):
        diff = predict - soft_y
        error = torch.abs(diff)
        return torch.mean(error)