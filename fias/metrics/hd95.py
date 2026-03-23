"""Approximate HD95 metric."""

from __future__ import annotations

import torch


def _foreground_points(mask: torch.Tensor) -> torch.Tensor:
    points = torch.nonzero(mask > 0, as_tuple=False).float()
    if points.numel() == 0:
        return mask.new_zeros((1, mask.dim()), dtype=torch.float32)
    return points


def hd95_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    scores = []
    for pred_mask, target_mask in zip(preds, target):
        pred_points = _foreground_points(pred_mask)
        target_points = _foreground_points(target_mask)
        dists = torch.cdist(pred_points, target_points, p=2)
        min_pred = dists.min(dim=1).values
        min_target = dists.min(dim=0).values
        combined = torch.cat([min_pred, min_target], dim=0)
        scores.append(torch.quantile(combined, 0.95).item())
    return float(sum(scores) / max(len(scores), 1))
