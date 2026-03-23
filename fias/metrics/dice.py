"""Dice metric."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> float:
    preds = torch.argmax(logits, dim=1)
    num_classes = logits.shape[1]
    pred_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (pred_one_hot * target_one_hot).sum(dims)
    union = pred_one_hot.sum(dims) + target_one_hot.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return float(dice.mean().item())
