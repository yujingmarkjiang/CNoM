from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cnom.models.common import ModelOutput


class AutoVideoClassifier(nn.Module):
    def __init__(self, pretrained_id: str, num_classes: int, pretrained: bool = True, **hub: Any) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModelForVideoClassification
        except ImportError as exc:
            raise RuntimeError("transformers required") from exc
        args = {k: v for k, v in hub.items() if v is not None}
        cfg = AutoConfig.from_pretrained(pretrained_id, num_labels=num_classes, **args)
        self.model = (
            AutoModelForVideoClassification.from_pretrained(pretrained_id, config=cfg, ignore_mismatched_sizes=True, **args)
            if pretrained
            else AutoModelForVideoClassification.from_config(cfg)
        )

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        out = self.model(pixel_values=video, output_hidden_states=True, return_dict=True)
        feat = out.hidden_states[-1].mean(1) if out.hidden_states else None
        return ModelOutput(logits=out.logits, features=feat)


class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.probe = nn.Linear(dim, num_classes)

    def forward(self, features: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        logits = self.probe(features)
        return ModelOutput(logits=logits, features=features)


class TimmImageSentinel(nn.Module):
    def __init__(self, architecture: str, num_classes: int, num_heads: int = 8, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm required") from exc
        self.backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
        dim = int(self.backbone.num_features)
        self.heads = nn.ModuleList([nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1)) for _ in range(num_heads)])
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        if images.ndim == 5:
            b, t, c, h, w = images.shape
            feat = self.backbone(images.reshape(b * t, c, h, w)).reshape(b, t, -1).mean(1)
        else:
            feat = self.backbone(images)
        unsafe = torch.cat([head(feat) for head in self.heads], dim=-1)
        safe = torch.zeros(feat.size(0), 1, device=feat.device)
        logits = torch.cat([safe, unsafe], dim=-1)
        return ModelOutput(logits=logits, features=feat)


class QwenVLClassifier(nn.Module):
    def __init__(self, pretrained_id: str, num_classes: int, **hub: Any) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers required") from exc
        args = {k: v for k, v in hub.items() if v is not None}
        self.processor = AutoProcessor.from_pretrained(pretrained_id, **args)
        self.model = AutoModelForVision2Seq.from_pretrained(pretrained_id, **args)
        hidden = int(getattr(self.model.config, "hidden_size", 4096))
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        feat = video.mean(dim=(1, 3, 4)) if video.ndim == 5 else video.mean(dim=(2, 3))
        if feat.size(-1) != self.head.in_features:
            feat = torch.nn.functional.adaptive_avg_pool1d(feat.unsqueeze(1), self.head.in_features).squeeze(1)
        return ModelOutput(logits=self.head(feat), features=feat)
