"""Context-aware fusion block."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextAwareFusion(nn.Module):
    def __init__(self, local_channels: int, global_channels: int, out_channels: int):
        super().__init__()
        merged_channels = local_channels + global_channels
        hidden_channels = max(out_channels // 2, 8)
        self.channel_reduce = nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=False)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([local_feat, global_feat], dim=1)
        fused = self.channel_reduce(fused)
        fused = fused * self.channel_gate(fused)
        fused = fused * self.spatial_gate(fused)
        return self.out_proj(fused)
