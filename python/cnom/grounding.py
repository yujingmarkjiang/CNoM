from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cnom.data.manifest import load_manifest, save_manifest
from cnom.data.video import VideoDecoder
from cnom.models.grounding import CocoGrounder, save_grounding_feature
from cnom.utils import resolve_device, safe_artifact_stem


def precompute_grounding(config: dict[str, Any], split: str | None = None) -> pd.DataFrame:
    dataset = config["dataset"]
    grounding = config["grounding"]
    frame = load_manifest(dataset["manifest"], root=dataset.get("root"), strict_paths=bool(dataset.get("strict_paths", True)))
    if split:
        selected = frame["split"].astype(str).eq(split)
    else:
        selected = pd.Series(True, index=frame.index)
    output_dir = Path(grounding.get("feature_dir", "artifacts/grounding"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(grounding.get("device", config.get("device", "auto"))))
    extractor = CocoGrounder(float(grounding.get("score_threshold", 0.35)), tuple(grounding.get("categories", ())))
    decoder = VideoDecoder(str(dataset.get("decoder", "opencv")))
    num_frames = int(grounding.get("sample_frames", 8))
    if num_frames <= 0:
        raise ValueError("grounding.sample_frames must be positive")
    for index, row in frame.loc[selected].iterrows():
        path = output_dir / f"{safe_artifact_stem(row.sample_id)}.npy"
        if path.is_file() and not bool(grounding.get("overwrite", False)):
            frame.loc[index, "object_features_path"] = str(path.resolve())
            continue
        video = decoder.decode(row.video_path, float(row.start_s), float(row.end_s), num_frames, "uniform")
        feature = extractor.extract(video.float().div(255.0), device)
        save_grounding_feature(feature, path)
        frame.loc[index, "object_features_path"] = str(path.resolve())
    save_manifest(frame, grounding.get("manifest_output", dataset["manifest"]))
    return frame
