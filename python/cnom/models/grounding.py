from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

from cnom.models.common import ModelOutput


class CocoGrounder:
    def __init__(self, score_threshold: float = 0.35, categories: Sequence[str] = ()) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories = list(weights.meta["categories"])
        self.selected = [str(name).replace("_", " ").strip() for name in categories]
        unknown = sorted(set(self.selected).difference(self.categories))
        if unknown:
            raise ValueError(f"Unknown COCO categories: {', '.join(unknown)}")
        if not self.selected:
            raise ValueError("At least one COCO category is required")
        self.indices = {name: self.categories.index(name) for name in self.selected}
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
        self.score_threshold = score_threshold
        self._device: torch.device | None = None

    @torch.inference_mode()
    def extract(self, video: torch.Tensor, device: torch.device) -> torch.Tensor:
        if video.ndim != 4 or video.shape[1] != 3 or not video.shape[0]:
            raise ValueError("video must have shape [T, 3, H, W] with at least one frame")
        if self._device != device:
            self.model.to(device)
            self._device = device
        frames = [frame.to(device, non_blocking=True) for frame in video]
        outputs = self.model(frames)
        vectors = [self._vector(output, frame.shape[-2], frame.shape[-1]).cpu() for output, frame in zip(outputs, video, strict=True)]
        stacked = torch.stack(vectors)
        summary = torch.cat([stacked.mean(0), stacked.max(0).values, stacked.std(0, unbiased=False)], dim=0)
        return summary

    def _vector(self, output: dict[str, torch.Tensor], height: int, width: int) -> torch.Tensor:
        vector = torch.zeros(max(1, len(self.selected)) * 3)
        scores, labels, boxes = output["scores"], output["labels"], output["boxes"]
        for pos, name in enumerate(self.selected):
            label = self.indices.get(name)
            if label is None:
                continue
            mask = (labels == label) & (scores >= self.score_threshold)
            if mask.any():
                selected_boxes = boxes[mask]
                areas = (selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1]) / (height * width)
                values = torch.stack([mask.sum().float(), scores[mask].max(), areas.max()]).detach().cpu()
                vector[pos * 3: pos * 3 + 3] = values
        return vector


class GroundedClassifier(nn.Module):
    def __init__(self, base: nn.Module, num_classes: int, object_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.base = base
        self.fusion = nn.Sequential(nn.LayerNorm(object_dim), nn.Linear(object_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        output = self.base(video)
        if object_features is None:
            return output
        delta = self.fusion(object_features)
        return ModelOutput(logits=output.logits + delta, features=output.features, frame_logits=output.frame_logits, segment_scores=output.segment_scores, extra=output.extra)


def save_grounding_feature(feature: torch.Tensor, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, feature.cpu().numpy().astype(np.float32))
