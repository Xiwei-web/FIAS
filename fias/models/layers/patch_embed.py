"""Overlap patch embedding."""

from __future__ import annotations

import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
