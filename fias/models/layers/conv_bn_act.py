"""Common convolution blocks."""

from __future__ import annotations

import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels)
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
