"""FIAS model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import MixAttentionDecoder
from .encoders import DMKEncoder, DilateFormerEncoder
from .fusion import ContextAwareFusion
from .heads import DeepSupervisionHead, SegmentationHead


class FIASModel(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 9, channels: tuple[int, int, int, int] = (32, 64, 128, 256)):
        super().__init__()
        self.global_encoder = DilateFormerEncoder(in_channels=in_channels, channels=channels)
        self.local_encoder = DMKEncoder(in_channels=in_channels, channels=channels)
        self.fusions = nn.ModuleList(
            [ContextAwareFusion(ch, ch, ch) for ch in channels]
        )
        self.decoder = MixAttentionDecoder(channels=channels)
        self.seg_head = SegmentationHead(channels[0], num_classes)
        self.deep_supervision_head = DeepSupervisionHead([channels[3], channels[2], channels[1]], num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        global_features = self.global_encoder(x)
        local_features = self.local_encoder(x)
        fused_features = [fusion(local_feat, global_feat) for fusion, local_feat, global_feat in zip(self.fusions, local_features, global_features)]
        decoder_out, decoder_features = self.decoder(fused_features)
        logits = self.seg_head(decoder_out)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        aux_outputs = self.deep_supervision_head(decoder_features[:3], x.shape[-2:])
        return {"logits": logits, "aux_logits": aux_outputs, "fused_features": fused_features}
