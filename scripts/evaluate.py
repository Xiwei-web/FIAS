"""Evaluation entrypoint for FIAS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fias.datasets import ACDCDataset, SynapseDataset, create_dataloader
from fias.datasets.transforms import default_eval_transforms
from fias.engine import Evaluator
from fias.models import FIASModel
from fias.utils import get_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FIAS.")
    parser.add_argument("--dataset", choices=["synapse", "acdc"], default="synapse")
    parser.add_argument("--data-root", type=Path, default=Path("./data/processed"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    logger = get_logger("fias.eval")
    dataset, num_classes = build_dataset(args)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    model = FIASModel(in_channels=args.in_channels, num_classes=num_classes)
    load_checkpoint(str(args.checkpoint), model, map_location=args.device)
    evaluator = Evaluator(model=model, device=args.device)
    metrics = evaluator.evaluate(dataloader)

    logger.info(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
