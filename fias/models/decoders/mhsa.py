"""Multi-head self-attention over 2D features."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialMHSA(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out
