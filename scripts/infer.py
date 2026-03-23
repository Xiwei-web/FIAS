"""Inference entrypoint for FIAS."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fias.datasets.transforms import default_eval_transforms
from fias.engine import Inferencer
from fias.models import FIASModel
from fias.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run FIAS inference on tensor files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image tensor .pt file.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to save predicted mask tensor.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    image = torch.load(args.input, map_location="cpu").float()
    sample = {"image": image, "mask": torch.zeros(image.shape[-2:], dtype=torch.long), "id": args.input.stem}
    image = default_eval_transforms(tuple(args.image_size))(sample)["image"].unsqueeze(0)

    model = FIASModel(in_channels=args.in_channels, num_classes=args.num_classes)
    load_checkpoint(str(args.checkpoint), model, map_location=args.device)

    inferencer = Inferencer(model=model, device=args.device)
    prediction = inferencer.predict(image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction.cpu(), args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
