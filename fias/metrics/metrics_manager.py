"""Metric manager."""

from __future__ import annotations

from .dice import dice_score
from .hd95 import hd95_score


class MetricsManager:
    def __call__(self, outputs, target):
        logits = outputs["logits"]
        return {"dice": dice_score(logits, target), "hd95": hd95_score(logits, target)}
