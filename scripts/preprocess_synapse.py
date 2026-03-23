"""Preprocess Synapse volumes into 2D tensor slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_nifti(path: Path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for preprocessing NIfTI files.") from exc
    return nib.load(str(path)).get_fdata()


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    image = image.clamp(min=image.quantile(0.01), max=image.quantile(0.99))
    mean = image.mean()
    std = image.std().clamp_min(1e-6)
    return (image - mean) / std


def preprocess_case(image_path: Path, label_path: Path, output_dir: Path) -> int:
    image_volume = torch.from_numpy(_load_nifti(image_path)).permute(2, 0, 1)
    label_volume = torch.from_numpy(_load_nifti(label_path)).permute(2, 0, 1).long()
    case_id = image_path.name.replace(".nii.gz", "").replace(".nii", "")

    saved = 0
    for slice_idx, (image_slice, label_slice) in enumerate(zip(image_volume, label_volume)):
        if torch.count_nonzero(label_slice) == 0:
            continue
        image_tensor = _normalize_image(image_slice.unsqueeze(0))
        mask_tensor = label_slice
        torch.save(image_tensor, output_dir / f"{case_id}_{slice_idx:03d}_image.pt")
        torch.save(mask_tensor, output_dir / f"{case_id}_{slice_idx:03d}_mask.pt")
        saved += 1
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Synapse dataset into 2D .pt files.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory with image NIfTI files.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory with label NIfTI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save .pt slices.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(args.images_dir.glob("*.nii.gz")) + list(args.images_dir.glob("*.nii")))
    total_saved = 0
    for image_path in image_paths:
        stem = image_path.name.replace(".nii.gz", "").replace(".nii", "")
        label_name = stem.replace("img", "label")
        label_path = args.labels_dir / f"{label_name}.nii.gz"
        if not label_path.exists():
            alt_label_path = args.labels_dir / f"{label_name}.nii"
            if alt_label_path.exists():
                label_path = alt_label_path
            else:
                continue
        total_saved += preprocess_case(image_path, label_path, args.output_dir)

    print(f"Saved {total_saved} slices to {args.output_dir}")


if __name__ == "__main__":
    main()
