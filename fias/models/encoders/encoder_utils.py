"""Encoder utilities."""

from __future__ import annotations

import torch.nn as nn

from ..layers import ConvBNAct


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride),
            ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.block(x)
