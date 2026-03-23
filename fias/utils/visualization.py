"""Visualization utilities."""

from __future__ import annotations

from pathlib import Path

import torch


def save_prediction_grid(prediction: torch.Tensor, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction.cpu(), path)
