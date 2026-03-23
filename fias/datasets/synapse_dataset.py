"""Synapse dataset definition."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .transforms import default_eval_transforms


def _load_tensor(path: Path) -> torch.Tensor:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported sample format: {path}")


class SynapseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        image_size: tuple[int, int] = (256, 256),
        num_classes: int = 9,
        synthetic_length: int = 8,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or default_eval_transforms(image_size)
        self.num_classes = num_classes
        self.samples = self._discover_samples(split_file)
        self.synthetic_length = synthetic_length

    def _discover_samples(self, split_file: str | Path | None) -> List[Dict[str, Path]]:
        if split_file is None:
            split_paths = sorted(self.root.glob("*_image.pt"))
            return [
                {"image": image_path, "mask": image_path.with_name(image_path.name.replace("_image.pt", "_mask.pt"))}
                for image_path in split_paths
                if image_path.with_name(image_path.name.replace("_image.pt", "_mask.pt")).exists()
            ]
        split_path = Path(split_file)
        if not split_path.exists():
            return []
        samples: List[Dict[str, Path]] = []
        for line in split_path.read_text().splitlines():
            case_id = line.strip()
            if not case_id:
                continue
            image_path = self.root / f"{case_id}_image.pt"
            mask_path = self.root / f"{case_id}_mask.pt"
            if image_path.exists() and mask_path.exists():
                samples.append({"image": image_path, "mask": mask_path})
        return samples

    def __len__(self) -> int:
        return len(self.samples) if self.samples else self.synthetic_length

    def _synthetic_sample(self, index: int) -> Dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        image = torch.randn(1, 256, 256, generator=generator)
        mask = torch.randint(0, self.num_classes, (256, 256), generator=generator)
        return {"image": image, "mask": mask, "id": f"synthetic_synapse_{index}"}

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            sample = self._synthetic_sample(index)
            return self.transform(sample) if self.transform else sample

        paths = self.samples[index]
        image = _load_tensor(paths["image"]).float()
        mask = _load_tensor(paths["mask"]).long()
        sample = {"image": image, "mask": mask, "id": paths["image"].stem.replace("_image", "")}
        return self.transform(sample) if self.transform else sample
