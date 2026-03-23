"""Lightweight transform utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, MutableMapping, Sequence

import torch
import torch.nn.functional as F


Sample = MutableMapping[str, torch.Tensor]


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Sample], Sample]]):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


@dataclass
class Resize2D:
    size: Sequence[int]

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"].float()
        mask = sample["mask"].float()
        sample["image"] = F.interpolate(image.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        sample["mask"] = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=self.size, mode="nearest").squeeze(0).squeeze(0).long()
        return sample


class NormalizeIntensity:
    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"].float()
        mean = image.mean()
        std = image.std().clamp_min(1e-6)
        sample["image"] = (image - mean) / std
        return sample


@dataclass
class RandomFlip:
    p: float = 0.5
    dims: Iterable[int] = (-1,)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        dims = tuple(self.dims)
        sample["image"] = torch.flip(sample["image"], dims=dims)
        sample["mask"] = torch.flip(sample["mask"], dims=dims)
        return sample


@dataclass
class RandomRotate90:
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        k = random.randint(0, 3)
        sample["image"] = torch.rot90(sample["image"], k=k, dims=(-2, -1))
        sample["mask"] = torch.rot90(sample["mask"], k=k, dims=(-2, -1))
        return sample


def default_train_transforms(image_size: Sequence[int]) -> Compose:
    return Compose([Resize2D(image_size), RandomFlip(), RandomRotate90(), NormalizeIntensity()])


def default_eval_transforms(image_size: Sequence[int]) -> Compose:
    return Compose([Resize2D(image_size), NormalizeIntensity()])
