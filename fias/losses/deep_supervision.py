"""Deep supervision loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, weights: list[float] | None = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, preds: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not preds:
            return target.new_tensor(0.0, dtype=torch.float32)
        weights = self.weights or [1.0 / len(preds)] * len(preds)
        loss = preds[0].new_tensor(0.0)
        for pred, weight in zip(preds, weights):
            loss = loss + weight * self.base_loss(pred, target)
        return loss
