from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import math


@dataclass(frozen=True)
class VideoPreprocessing:
    image_size: int
    num_frames: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULTS: dict[str, VideoPreprocessing] = {
    "mobilenetv3_sentinel": VideoPreprocessing(224, 32, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "video_swin_t": VideoPreprocessing(224, 32, (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
    "x3d_xs": VideoPreprocessing(182, 13, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    "x3d_s": VideoPreprocessing(182, 13, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    "x3d_m": VideoPreprocessing(224, 16, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    "slowfast_r50": VideoPreprocessing(256, 32, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    "vivit": VideoPreprocessing(224, 32, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "videomae": VideoPreprocessing(224, 16, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "timesformer": VideoPreprocessing(224, 8, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "vadclip": VideoPreprocessing(224, 32, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}


def resolve_video_preprocessing(model: dict[str, Any], dataset: dict[str, Any] | None = None, allow_remote: bool = True) -> VideoPreprocessing:
    dataset = dataset or {}
    name = str(model.get("name", "")).lower()
    fallback = _DEFAULTS.get(name, VideoPreprocessing(224, 32, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    processor = _processor_values(str(model.get("pretrained_id", "")), bool(model.get("processor_local_files_only", False)), allow_remote) if name in {"vivit", "videomae", "timesformer"} else None
    processor_size, processor_mean, processor_std = processor or (None, None, None)
    image_size = int(model.get("image_size", dataset.get("image_size", processor_size or fallback.image_size)))
    num_frames = int(model.get("num_frames", dataset.get("num_frames", fallback.num_frames)))
    mean = _triple(model.get("mean", dataset.get("mean", processor_mean or fallback.mean)), "mean")
    std = _triple(model.get("std", dataset.get("std", processor_std or fallback.std)), "std")
    if image_size <= 0 or num_frames <= 0:
        raise ValueError("Preprocessing image_size and num_frames must be positive")
    if not all(math.isfinite(value) for value in (*mean, *std)):
        raise ValueError("Preprocessing statistics must be finite")
    if any(value <= 0 for value in std):
        raise ValueError("Preprocessing standard deviations must be positive")
    return VideoPreprocessing(image_size, num_frames, mean, std)


@lru_cache(maxsize=16)
def _processor_values(pretrained_id: str, local_files_only: bool, allow_remote: bool) -> tuple[int | None, tuple[float, float, float] | None, tuple[float, float, float] | None] | None:
    if not pretrained_id:
        return None
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(pretrained_id, local_files_only=local_files_only or not allow_remote)
    except Exception:
        return None
    size = getattr(processor, "crop_size", None) or getattr(processor, "size", None)
    if isinstance(size, dict):
        size = size.get("shortest_edge") or size.get("height") or size.get("width")
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    return (int(size) if size else None, _triple(mean, "image_mean") if mean is not None else None, _triple(std, "image_std") if std is not None else None)


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value), float(value)
    if value is None:
        raise ValueError(f"{name} must contain three channel values")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three channel values")
    return values
