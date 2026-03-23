"""Checkpoint IO."""

from __future__ import annotations

import torch


def save_checkpoint(path: str, model, optimizer=None, **extra):
    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, map_location: str = "cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
