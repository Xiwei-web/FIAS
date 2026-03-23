"""Evaluator."""

from __future__ import annotations

import torch

from ..metrics import MetricsManager


class Evaluator:
    def __init__(self, model, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.metrics = MetricsManager()

    def evaluate(self, dataloader):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
                pred = self.model(batch["image"])
                outputs.append(self.metrics(pred, batch["mask"]))
        if not outputs:
            return {"dice": 0.0, "hd95": 0.0}
        return {
            "dice": sum(item["dice"] for item in outputs) / len(outputs),
            "hd95": sum(item["hd95"] for item in outputs) / len(outputs),
        }
