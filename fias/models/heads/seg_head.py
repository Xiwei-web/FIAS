"""Segmentation output head."""

from __future__ import annotations

import torch.nn as nn


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)
