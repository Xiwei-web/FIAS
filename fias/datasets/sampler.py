"""DataLoader helpers."""

from __future__ import annotations

from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size: int = 2, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
