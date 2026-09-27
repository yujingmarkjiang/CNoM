from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cnom.data.manifest import load_manifest, save_manifest
from cnom.data.video import VideoDecoder, VideoTransform
from cnom.models.preprocessing import resolve_video_preprocessing
from cnom.models.registry import build_model
from cnom.utils import autocast_dtype, resolve_device, safe_artifact_stem


@torch.inference_mode()
def extract_feature_bags(config: dict[str, Any], split: str | None = None) -> pd.DataFrame:
    dataset, feature_cfg = config["dataset"], config["features"]
    manifest = load_manifest(dataset["manifest"], root=dataset.get("root"), strict_paths=bool(dataset.get("strict_paths", True)))
    selected = manifest["split"].astype(str).eq(split) if split else pd.Series(True, index=manifest.index)
    model_cfg = dict(feature_cfg["backbone"])
    model = build_model(model_cfg, checkpoint=model_cfg.get("checkpoint"), strict=False)
    device = resolve_device(str(feature_cfg.get("device", config.get("device", "auto"))))
    precision, dtype = str(feature_cfg.get("precision", config.get("precision", "fp32"))), None
    dtype = autocast_dtype(precision)
    model.to(device).eval()
    decoder = VideoDecoder(str(dataset.get("decoder", "opencv")))
    preprocessing = resolve_video_preprocessing(model_cfg, dataset)
    transform = VideoTransform(preprocessing.image_size, train=False, mean=preprocessing.mean, std=preprocessing.std)
    segments, frames_per_segment = int(feature_cfg.get("num_segments", 32)), preprocessing.num_frames
    output_dir = Path(feature_cfg.get("output_dir", "artifacts/features"))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(feature_cfg.get("batch_size", 8))
    if segments <= 0 or batch_size <= 0:
        raise ValueError("features.num_segments and features.batch_size must be positive")
    for index, row in manifest.loc[selected].iterrows():
        output_path = output_dir / f"{safe_artifact_stem(row.sample_id)}.npy"
        if output_path.is_file() and not bool(feature_cfg.get("overwrite", False)):
            manifest.loc[index, "feature_path"] = str(output_path.resolve())
            continue
        boundaries = np.linspace(float(row.start_s), float(row.end_s), segments + 1)
        bags: list[np.ndarray] = []
        videos: list[torch.Tensor] = []
        for start_s, end_s in zip(boundaries[:-1], boundaries[1:], strict=True):
            videos.append(transform(decoder.decode(row.video_path, float(start_s), float(end_s), frames_per_segment, "uniform")))
            if len(videos) == batch_size:
                bags.extend(_feature_batch(model, videos, device, dtype))
                videos = []
        if videos:
            bags.extend(_feature_batch(model, videos, device, dtype))
        np.save(output_path, np.stack(bags).astype(np.float32))
        manifest.loc[index, "feature_path"] = str(output_path.resolve())
    save_manifest(manifest, feature_cfg.get("manifest_output", dataset["manifest"]))
    return manifest


def _feature_batch(model, videos, device, dtype) -> list[np.ndarray]:
    batch = torch.stack(videos).to(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type != "cpu"):
        output = model(batch)
    if output.features is None:
        raise RuntimeError(f"{model.__class__.__name__} does not expose features")
    return [value for value in output.features.detach().cpu().numpy()]
