"""Depthwise multi-kernel encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers import ConvBNAct
from .encoder_utils import DownsampleStage


class DMKBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.local = ConvBNAct(in_channels, out_channels, kernel_size=3)
        self.dw5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=False)
        self.dw7 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.proj = ConvBNAct(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        mixed = local + self.dw5(local) + self.dw7(local)
        mixed = self.bn(mixed)
        return self.proj(mixed)


class DMKEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels: tuple[int, int, int, int] = (32, 64, 128, 256)):
        super().__init__()
        self.stem = ConvBNAct(in_channels, channels[0], kernel_size=3)
        stages = []
        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            stages.append(DownsampleStage(in_ch, out_ch, stride=1 if in_ch == out_ch else 2))
            blocks.append(DMKBlock(out_ch, out_ch))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage, block in zip(self.stages, self.blocks):
            x = stage(x)
            x = block(x)
            features.append(x)
        return features
