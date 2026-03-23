"""Monte Carlo attention approximation."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloAttention(nn.Module):
    def __init__(self, pool_sizes: tuple[int, int, int] = (3, 2, 1)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.gate = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = []
        base_size = x.shape[-2:]
        mean_map = x.mean(dim=1, keepdim=True)
        for size in self.pool_sizes:
            if size == 1:
                pooled_map = F.adaptive_avg_pool2d(mean_map, output_size=1)
            else:
                pooled_map = F.adaptive_avg_pool2d(mean_map, output_size=size)
            pooled_map = F.interpolate(pooled_map, size=base_size, mode="bilinear", align_corners=False)
            pooled.append(pooled_map)
        attention = random.choice(pooled) if self.training else sum(pooled) / len(pooled)
        return self.gate(attention)
