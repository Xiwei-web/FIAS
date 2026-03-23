"""ACDC dataset definition."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .transforms import default_eval_transforms


class ACDCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        image_size: tuple[int, int] = (256, 256),
        num_classes: int = 4,
        synthetic_length: int = 8,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or default_eval_transforms(image_size)
        self.num_classes = num_classes
        self.samples = self._discover_samples(split_file)
        self.synthetic_length = synthetic_length

    def _discover_samples(self, split_file: str | Path | None) -> List[Dict[str, Path]]:
        if split_file is None:
            image_paths = sorted(self.root.glob("*_image.pt"))
            return [
                {"image": image_path, "mask": image_path.with_name(image_path.name.replace("_image.pt", "_mask.pt"))}
                for image_path in image_paths
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            generator = torch.Generator().manual_seed(10_000 + index)
            sample = {
                "image": torch.randn(1, 256, 256, generator=generator),
                "mask": torch.randint(0, self.num_classes, (256, 256), generator=generator),
                "id": f"synthetic_acdc_{index}",
            }
            return self.transform(sample) if self.transform else sample

        paths = self.samples[index]
        sample = {
            "image": torch.load(paths["image"], map_location="cpu").float(),
            "mask": torch.load(paths["mask"], map_location="cpu").long(),
            "id": paths["image"].stem.replace("_image", ""),
        }
        return self.transform(sample) if self.transform else sample
