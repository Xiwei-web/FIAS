"""Training entrypoint for FIAS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fias.datasets import ACDCDataset, SynapseDataset, create_dataloader
from fias.datasets.transforms import default_train_transforms
from fias.engine import Trainer
from fias.losses import FeatureMixingSegLoss
from fias.models import FIASModel
from fias.utils import get_logger, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train FIAS.")
    parser.add_argument("--dataset", choices=["synapse", "acdc"], default="synapse")
    parser.add_argument("--data-root", type=Path, default=Path("./data/processed"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/checkpoints"))
    return parser.parse_args()


def build_dataset(args):
    image_size = tuple(args.image_size)
    if args.dataset == "synapse":
        num_classes = args.num_classes or 9
        return SynapseDataset(
            root=args.data_root,
            split_file=args.split_file,
            transform=default_train_transforms(image_size),
            image_size=image_size,
            num_classes=num_classes,
        ), num_classes
    num_classes = args.num_classes or 4
    return ACDCDataset(
        root=args.data_root,
        split_file=args.split_file,
        transform=default_train_transforms(image_size),
        image_size=image_size,
        num_classes=num_classes,
    ), num_classes


def main():
    args = parse_args()
    logger = get_logger("fias.train")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, num_classes = build_dataset(args)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    model = FIASModel(in_channels=args.in_channels, num_classes=num_classes)
    criterion = FeatureMixingSegLoss(gamma=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=args.device)

    history = trainer.fit(dataloader, epochs=args.epochs)
    logger.info("Training finished.")
    logger.info(json.dumps(history, ensure_ascii=False, indent=2))

    save_checkpoint(str(args.output_dir / "last.pt"), model, optimizer=optimizer, history=history, args=vars(args))


if __name__ == "__main__":
    main()
