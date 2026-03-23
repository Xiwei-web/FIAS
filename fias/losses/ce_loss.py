"""Cross entropy loss wrapper."""

from __future__ import annotations

import torch.nn as nn


class CrossEntropySegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.loss(logits, target.long())
