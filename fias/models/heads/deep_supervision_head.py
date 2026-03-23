"""Deep supervision heads."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels_list: list[int], num_classes: int):
        super().__init__()
        self.heads = nn.ModuleList([nn.Conv2d(ch, num_classes, kernel_size=1) for ch in in_channels_list])

    def forward(self, features, output_size):
        outputs = []
        for feature, head in zip(features, self.heads):
            pred = head(feature)
            pred = F.interpolate(pred, size=output_size, mode="bilinear", align_corners=False)
            outputs.append(pred)
        return outputs
