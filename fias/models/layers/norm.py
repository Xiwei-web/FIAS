"""Normalization helpers."""

from __future__ import annotations

import torch.nn as nn


def build_norm_2d(num_features: int, norm_type: str = "bn") -> nn.Module:
    if norm_type == "bn":
        return nn.BatchNorm2d(num_features)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_features)
    raise ValueError(f"Unsupported norm type: {norm_type}")
