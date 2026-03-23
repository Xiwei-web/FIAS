"""Upsampling blocks."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .conv_bn_act import ConvBNAct


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBNAct(in_channels, out_channels, kernel_size=3)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)
