"""Feature mixing loss used by FIAS."""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn

from .ce_loss import CrossEntropySegLoss
from .deep_supervision import DeepSupervisionLoss
from .dice_loss import DiceLoss


class FeatureMixingSegLoss(nn.Module):
    def __init__(self, gamma: float = 0.4):
        super().__init__()
        self.gamma = gamma
        self.dice = DiceLoss()
        self.ce = CrossEntropySegLoss()
        self.deep_supervision = DeepSupervisionLoss(self._single_loss)

    def _single_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.dice(logits, target) + (1.0 - self.gamma) * self.ce(logits, target)

    def _mixed_predictions(self, aux_logits: list[torch.Tensor]) -> list[torch.Tensor]:
        if not aux_logits:
            return []
        mixed = []
        total = len(aux_logits)
        for subset_size in range(1, total + 1):
            for subset in combinations(aux_logits, subset_size):
                mixed.append(torch.stack(list(subset), dim=0).mean(dim=0))
        return mixed

    def forward(self, outputs: dict[str, torch.Tensor | list[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        logits = outputs["logits"]
        aux_logits = outputs.get("aux_logits", [])
        loss = self._single_loss(logits, target)
        mixed_preds = self._mixed_predictions(aux_logits)
        if mixed_preds:
            loss = loss + self.deep_supervision(mixed_preds, target)
        return loss
