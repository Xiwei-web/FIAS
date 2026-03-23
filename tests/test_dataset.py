from pathlib import Path

import torch

from fias.datasets import ACDCDataset, SynapseDataset
from fias.datasets.transforms import default_eval_transforms


def test_synapse_dataset_synthetic_sample():
    dataset = SynapseDataset(root=".", split_file=None, transform=default_eval_transforms((128, 128)), synthetic_length=2)

    sample = dataset[0]

    assert sample["image"].shape == (1, 128, 128)
    assert sample["mask"].shape == (128, 128)
    assert sample["mask"].dtype == torch.long
    assert sample["id"].startswith("synthetic_synapse_")


def test_acdc_dataset_synthetic_sample():
    dataset = ACDCDataset(root=".", split_file=None, transform=default_eval_transforms((96, 96)), synthetic_length=2)

    sample = dataset[1]

    assert sample["image"].shape == (1, 96, 96)
    assert sample["mask"].shape == (96, 96)
    assert sample["mask"].dtype == torch.long
    assert sample["id"].startswith("synthetic_acdc_")


def test_synapse_dataset_reads_pt_files(tmp_path: Path):
    image = torch.randn(1, 64, 64)
    mask = torch.randint(0, 9, (64, 64))
    torch.save(image, tmp_path / "case001_image.pt")
    torch.save(mask, tmp_path / "case001_mask.pt")

    dataset = SynapseDataset(root=tmp_path, transform=default_eval_transforms((64, 64)))
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["id"] == "case001"
    assert sample["image"].shape == (1, 64, 64)
    assert sample["mask"].shape == (64, 64)
