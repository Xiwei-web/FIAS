"""Inference helper."""

from __future__ import annotations

import torch


class Inferencer:
    def __init__(self, model, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
        return torch.argmax(outputs["logits"], dim=1)
