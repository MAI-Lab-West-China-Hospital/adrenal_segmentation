import glob

from scipy import ndimage
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class weight_map():
    def __init__(self, target):
        # target N,c=1,x,y,z
        self.target = target.cpu()
        self.batch_size = target.shape[0]
        self.H = target.shape[2]
        self.W = target.shape[3]
        self.D = target.shape[4]
        self.num = self.H * self.W * self.D

    def get_edge_points(self):
        """
        get edge points of a binary segmentation result
        return N,c=1,x,y,z, only boundary=1
        """
        dim = len(self.target[2:].shape)
        boundary = torch.zeros_like(self.target)  # n,1,x,y,z
        if (dim == 2):
            strt = ndimage.generate_binary_structure(2, 1)
        else:
            strt = ndimage.generate_binary_structure(3, 1)
        for i in range(self.batch_size):
            ero = ndimage.morphology.binary_erosion(self.target[i, 0], strt)
            edge = np.asarray(self.target[i, 0], np.uint8) - np.asarray(ero, np.uint8)
            # dilation
            # edge = ndimage.morphology.binary_dilation(edge, strt)
            boundary[i, 0] = torch.tensor(edge)
        return boundary

    def spatial_weight(self):
        boundary = self.get_edge_points()
        weights = torch.ones_like(self.target)
        for i in range(self.batch_size):
            bound_w = torch.sum(self.target[i, 0]) / torch.sum(boundary[i, 0])
            weights[i, 0][boundary[i, 0] == 1] = bound_w.to(weights)
        return weights.to(device)


class weight_FocalLoss(nn.Module):

    def __init__(
            self,
            gamma: float = 2.0,
            reduction: str = 'mean',
            weight: str = True
    ) -> None:

        super(weight_FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: the shape should be BCH[WD].
                where C (greater than 1) is the number of classes.
                Softmax over the logits is integrated in this module for improved numerical stability.
            target: the shape should be B1H[WD] or BCH[WD].
                If the target's shape is B1H[WD], the target that this loss expects should be a class index
                in the range [0, C-1] where C is the number of classes
        """
        i = logits
        t = target

        if i.ndimension() != t.ndimension():
            raise ValueError(f"logits and target ndim must match, got logits={i.ndimension()} target={t.ndimension()}.")

        if t.shape[1] != 1 and t.shape[1] != i.shape[1]:
            raise ValueError(
                "target must have one channel or have the same shape as the logits. "
                "If it has one channel, it should be a class index in the range [0, C-1] "
                f"where C is the number of classes inferred from 'logits': C={i.shape[1]}. "
            )
        if i.shape[1] == 1:
            raise NotImplementedError("Single-channel predictions not supported.")

        # Change the shape of logits and target to
        # num_batch x num_class x num_voxels.
        if i.dim() > 2:
            i = i.view(i.size(0), i.size(1), -1)  # N,C,H,W => N,C,H*W
            t = t.reshape(t.size(0), t.size(1), -1)  # N,1,H,W => N,1,H*W or N,C,H*W
        else:  # Compatibility with classification.
            i = i.unsqueeze(2)  # N,C => N,C,1
            t = t.unsqueeze(2)  # N,1 => N,1,1 or N,C,1

        # Compute the log proba (more stable numerically than softmax).
        logpt = F.log_softmax(i, dim=1)  # N,C,H*W
        # Keep only log proba values of the ground truth class for each voxel.
        if target.shape[1] == 1:
            logpt = logpt.gather(1, t.long())  # N,C,H*W => N,1,H*W
            logpt = torch.squeeze(logpt, dim=1)  # N,1,H*W => N,H*W

        # Get the proba
        pt = torch.exp(logpt)  # N,H*W or N,C,H*W

        if self.weight:
            weightmap = weight_map(target).spatial_weight()  # N,1,H,W, D
            weightmap = weightmap.reshape(weightmap.size(0), 1, -1)  # N,1,H*W*D
            # Convert the weight to a map in which each voxel
            # has the weight associated with the ground-truth label
            # associated with this voxel in target.
            at = torch.squeeze(weightmap, dim=1)   # N,1,H*W*D => N,H*W*D  .to(torch.device('cuda'))
            # Multiply the log proba by their weights.
            logpt = logpt * at

        # Compute the loss mini-batch.
        weight = torch.pow(-pt + 1.0, self.gamma)
        if target.shape[1] == 1:
            loss = torch.mean(-weight * logpt, dim=1)  # N
        else:
            loss = torch.mean(-weight * t * logpt, dim=-1)  # N,C

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return loss.mean()
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
