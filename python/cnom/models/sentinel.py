from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_large, mobilenet_v3_small

from cnom.models.common import ModelOutput


class MobileNetFeature(nn.Module):
    def __init__(self, variant: str, pretrained: bool, dropout: float) -> None:
        super().__init__()
        if variant == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = mobilenet_v3_small(weights=weights, dropout=dropout)
        elif variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = mobilenet_v3_large(weights=weights, dropout=dropout)
        else:
            raise ValueError(f"Unknown MobileNetV3 variant: {variant}")
        self.features = model.features
        self.pool = model.avgpool
        self.output_dim = int(model.classifier[0].in_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(images)).flatten(1)


class MobileNetV3Sentinel(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        num_heads: int = 8,
        variant: str = "large",
        pretrained: bool = True,
        dropout: float = 0.2,
        shared_backbone: bool = False,
        aggregation: str = "max",
        lse_temperature: float = 8.0,
        head_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_classes != num_heads + 1:
            raise ValueError("Sentinel uses one safe class plus one specialized head per unsafe class")
        if aggregation not in {"mean", "max", "lse"}:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        if aggregation == "lse" and lse_temperature <= 0:
            raise ValueError("lse_temperature must be positive")
        if head_hidden_dim < 0 or not 0 <= dropout < 1:
            raise ValueError("head_hidden_dim must be non-negative and dropout must be in [0, 1)")
        base = MobileNetFeature(variant, pretrained, dropout)
        self.shared_backbone = shared_backbone
        self.backbones = nn.ModuleList([base] if shared_backbone else [deepcopy(base) for _ in range(num_heads)])
        def make_head() -> nn.Sequential:
            if head_hidden_dim > 0:
                return nn.Sequential(nn.Linear(base.output_dim, head_hidden_dim), nn.Hardswish(), nn.Dropout(dropout), nn.Linear(head_hidden_dim, 1))
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(base.output_dim, 1))
        self.unsafe_heads = nn.ModuleList([make_head() for _ in range(num_heads)])
        self.safe_head = make_head()
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.lse_temperature = lse_temperature

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        if video.ndim != 5:
            raise ValueError("Sentinel input must be [batch, frames, channels, height, width]")
        batch, frames, channels, height, width = video.shape
        images = video.reshape(batch * frames, channels, height, width)
        if self.shared_backbone:
            shared = self.backbones[0](images)
            head_features = [shared] * len(self.unsafe_heads)
        else:
            head_features = [backbone(images) for backbone in self.backbones]
        unsafe = torch.cat([head(features) for head, features in zip(self.unsafe_heads, head_features, strict=True)], dim=1)
        pooled = torch.stack(head_features).mean(0)
        safe = self.safe_head(pooled)
        frame_logits = torch.cat([safe, unsafe], dim=1).reshape(batch, frames, self.num_classes)
        event_logits = self.aggregate_event_logits(frame_logits)
        features = pooled.reshape(batch, frames, -1).mean(1)
        return ModelOutput(logits=event_logits, features=features, frame_logits=frame_logits)

    def aggregate_event_logits(self, frame_logits: torch.Tensor) -> torch.Tensor:
        safe = frame_logits[:, :, :1].mean(1)
        unsafe = frame_logits[:, :, 1:]
        if self.aggregation == "mean":
            unsafe_event = unsafe.mean(1)
        elif self.aggregation == "max":
            unsafe_event = unsafe.max(1).values
        elif self.aggregation == "lse":
            temperature = self.lse_temperature
            unsafe_event = torch.logsumexp(unsafe * temperature, dim=1) / temperature
            unsafe_event = unsafe_event - torch.log(torch.tensor(unsafe.shape[1], device=unsafe.device, dtype=unsafe.dtype)) / temperature
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        return torch.cat([safe, unsafe_event], dim=1)

    @staticmethod
    def unsafe_scores(frame_logits: torch.Tensor) -> torch.Tensor:
        return frame_logits[..., 1:].sigmoid().amax(-1)
