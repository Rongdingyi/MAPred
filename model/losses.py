import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    class_weight, 
    pose_weight,
    mask: torch.Tensor = None,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", weight=class_weight, pos_weight=pose_weight
        )
        # print(ce_loss.mean())
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if mask is not None:
            loss = torch.einsum("bfn,bf->bfn", loss, mask)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss