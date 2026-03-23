"""Training hooks."""

from __future__ import annotations


class HookBase:
    def before_epoch(self, trainer):
        return None

    def after_epoch(self, trainer, metrics):
        return None
