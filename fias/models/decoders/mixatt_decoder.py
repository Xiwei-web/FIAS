"""Mixing attention decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvBNAct
from .mhsa import SpatialMHSA
from .monte_carlo_attention import MonteCarloAttention


class MixAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_heads: int):
        super().__init__()
        self.pre = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            nn.Dropout2d(0.1),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.mhsa = SpatialMHSA(out_channels, num_heads=num_heads)
        self.mca = MonteCarloAttention()
        self.out_proj = ConvBNAct(out_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.pre(x)
        self_attn = self.mhsa(x)
        mca = self.mca(x)
        mixed = x + self_attn + (x * mca)
        return self.out_proj(mixed)


class MixAttentionDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 256), heads: tuple[int, int, int, int] = (2, 4, 4, 8)):
        super().__init__()
        self.bridge = ConvBNAct(channels[-1], channels[-1], kernel_size=3)
        self.blocks = nn.ModuleList(
            [
                MixAttentionBlock(channels[3], channels[2], channels[2], heads[2]),
                MixAttentionBlock(channels[2], channels[1], channels[1], heads[1]),
                MixAttentionBlock(channels[1], channels[0], channels[0], heads[0]),
            ]
        )

    def forward(self, fused_features: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.bridge(fused_features[-1])
        decoder_features = [x]
        for block, skip in zip(self.blocks, reversed(fused_features[:-1])):
            x = block(x, skip)
            decoder_features.append(x)
        return x, decoder_features
