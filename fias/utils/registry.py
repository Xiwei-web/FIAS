"""Minimal registry."""

from __future__ import annotations


class Registry(dict):
    def register(self, name: str, value):
        self[name] = value
        return value
