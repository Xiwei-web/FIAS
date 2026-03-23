"""Simple trainer."""

from __future__ import annotations

from typing import Iterable

import torch

from ..metrics import MetricsManager


class Trainer:
    def __init__(self, model, optimizer, criterion, device: str = "cpu", hooks: Iterable | None = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.hooks = list(hooks or [])
        self.metrics = MetricsManager()

    def _move_batch(self, batch):
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        last_metrics = {"dice": 0.0, "hd95": 0.0}
        for batch in dataloader:
            batch = self._move_batch(batch)
            outputs = self.model(batch["image"])
            loss = self.criterion(outputs, batch["mask"])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            total_steps += 1
            last_metrics = self.metrics(outputs, batch["mask"])
        return {"loss": total_loss / max(total_steps, 1), **last_metrics}

    def fit(self, dataloader, epochs: int = 1):
        history = []
        for _ in range(epochs):
            for hook in self.hooks:
                hook.before_epoch(self)
            metrics = self.train_one_epoch(dataloader)
            for hook in self.hooks:
                hook.after_epoch(self, metrics)
            history.append(metrics)
        return history
