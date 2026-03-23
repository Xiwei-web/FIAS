"""Segmentation postprocess."""

from __future__ import annotations

import torch


def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)
