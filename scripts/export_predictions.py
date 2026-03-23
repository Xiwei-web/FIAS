"""Batch export predictions for preprocessed tensor slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fias.datasets import ACDCDataset, SynapseDataset, create_dataloader
from fias.datasets.transforms import default_eval_transforms
from fias.engine import Inferencer
from fias.models import FIASModel
from fias.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export predictions for a dataset.")
    parser.add_argument("--dataset", choices=["synapse", "acdc"], default="synapse")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_dataset(args):
    image_size = tuple(args.image_size)
    if args.dataset == "synapse":
        num_classes = args.num_classes or 9
        dataset = SynapseDataset(
            root=args.data_root,
            split_file=args.split_file,
            transform=default_eval_transforms(image_size),
            image_size=image_size,
            num_classes=num_classes,
        )
    else:
        num_classes = args.num_classes or 4
        dataset = ACDCDataset(
            root=args.data_root,
            split_file=args.split_file,
            transform=default_eval_transforms(image_size),
            image_size=image_size,
            num_classes=num_classes,
        )
    return dataset, num_classes


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, num_classes = build_dataset(args)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    model = FIASModel(in_channels=args.in_channels, num_classes=num_classes)
    load_checkpoint(str(args.checkpoint), model, map_location=args.device)
    inferencer = Inferencer(model=model, device=args.device)

    sample_index = 0
    for batch in dataloader:
        images = batch["image"]
        predictions = inferencer.predict(images)
        ids = batch.get("id")
        batch_size = predictions.shape[0]
        for idx in range(batch_size):
            sample_id = ids[idx] if ids is not None else f"sample_{sample_index:05d}"
            torch.save(predictions[idx].cpu(), args.output_dir / f"{sample_id}_pred.pt")
            sample_index += 1

    print(f"Exported predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
