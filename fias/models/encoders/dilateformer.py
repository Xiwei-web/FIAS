"""Practical DilateFormer-style encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers import ConvBNAct, OverlapPatchEmbed


class DilatedAttentionBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.pwconv = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        return x + residual


class TransformerStyleBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x_norm = self.norm(x).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        x = residual + attn_out
        return x + self.ffn(x)


class DilateFormerEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels: tuple[int, int, int, int] = (32, 64, 128, 256)):
        super().__init__()
        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(in_channels, channels[0], patch_size=7, stride=2),
                OverlapPatchEmbed(channels[0], channels[1], patch_size=3, stride=2),
                OverlapPatchEmbed(channels[1], channels[2], patch_size=3, stride=2),
                OverlapPatchEmbed(channels[2], channels[3], patch_size=3, stride=2),
            ]
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(DilatedAttentionBlock(channels[0], 1), DilatedAttentionBlock(channels[0], 2)),
                nn.Sequential(DilatedAttentionBlock(channels[1], 1), DilatedAttentionBlock(channels[1], 2)),
                nn.Sequential(TransformerStyleBlock(channels[2], 4), TransformerStyleBlock(channels[2], 4)),
                nn.Sequential(TransformerStyleBlock(channels[3], 8), TransformerStyleBlock(channels[3], 8)),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for embed, block in zip(self.patch_embeds, self.blocks):
            x = embed(x)
            x = block(x)
            features.append(x)
        return features
