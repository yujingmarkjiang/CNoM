from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cnom.utils import read_table, write_table

REQUIRED_COLUMNS = ["sample_id", "source_id", "video_path", "start_s", "end_s", "label", "split"]
OPTIONAL_COLUMNS = [
    "camera_id", "scene_id", "event_id", "cluster_id", "timestamp", "worker_proxy", "object_features_path",
    "machine_state_path", "feature_path", "frame_labels_path", "temporal_annotations", "visibility", "ambiguity",
    "adjudication_status", "is_hard_negative",
]


@dataclass(frozen=True)
class LeakageAudit:
    overlap_by_field: dict[str, int]
    invalid_intervals: int
    missing_files: int
    duplicated_sample_ids: int

    @property
    def passed(self) -> bool:
        return not any(self.overlap_by_field.values()) and self.invalid_intervals == 0 and self.missing_files == 0 and self.duplicated_sample_ids == 0


def load_manifest(path: str | Path, root: str | Path | None = None, strict_paths: bool = True) -> pd.DataFrame:
    frame = read_table(path).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest lacks required columns: {missing}")
    frame["label"] = frame["label"].astype(int)
    frame["start_s"] = frame["start_s"].astype(float)
    frame["end_s"] = frame["end_s"].astype(float)
    root_path = Path(root).resolve() if root else Path(path).resolve().parent
    frame["video_path"] = frame["video_path"].map(lambda value: str(_resolve_path(value, root_path)))
    for column in ("object_features_path", "feature_path", "frame_labels_path", "machine_state_path"):
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: str(_resolve_path(value, root_path)) if pd.notna(value) and str(value).strip() else None)
    validate_manifest(frame, strict_paths=strict_paths)
    return frame


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def validate_manifest(frame: pd.DataFrame, strict_paths: bool = True, labels: Iterable[int] | None = None) -> None:
    if frame.empty:
        raise ValueError("Manifest is empty")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be present and unique")
    if (frame["end_s"] <= frame["start_s"]).any() or (frame["start_s"] < 0).any():
        raise ValueError("Every clip must satisfy 0 <= start_s < end_s")
    if (frame["label"] < 0).any():
        raise ValueError("Labels must be non-negative integers")
    if frame["split"].isna().any() or frame["split"].astype(str).str.strip().eq("").any():
        raise ValueError("Every sample must have a non-empty split")
    if labels is not None and not set(frame["label"]).issubset(set(labels)):
        raise ValueError("Manifest contains labels outside the configured ontology")
    if strict_paths:
        missing = [value for value in frame["video_path"].unique() if not Path(value).is_file()]
        if missing:
            preview = missing[:5]
            raise FileNotFoundError(f"Manifest references {len(missing)} missing videos; examples: {preview}")


def audit_split_leakage(frame: pd.DataFrame, grouped_fields: Iterable[str]) -> LeakageAudit:
    overlaps: dict[str, int] = {}
    for field in grouped_fields:
        if field not in frame.columns:
            continue
        groups = frame.dropna(subset=[field]).groupby(field)["split"].nunique()
        overlaps[field] = int((groups > 1).sum())
    invalid = int(((frame["end_s"] <= frame["start_s"]) | (frame["start_s"] < 0)).sum())
    missing = int(sum(not Path(path).is_file() for path in frame["video_path"].unique()))
    duplicates = int(frame["sample_id"].duplicated().sum())
    return LeakageAudit(overlaps, invalid, missing, duplicates)


def save_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    columns = REQUIRED_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in columns]
    write_table(frame[columns + extras], path)


def load_stream_manifest(dataset_config: dict, strict_paths: bool | None = None) -> pd.DataFrame:
    """Return one full recording interval per source for temporal replay."""
    strict = bool(dataset_config.get("strict_paths", True)) if strict_paths is None else strict_paths
    configured = dataset_config.get("stream_manifest")
    if configured:
        return load_manifest(configured, root=dataset_config.get("root"), strict_paths=strict)
    frame = load_manifest(dataset_config["manifest"], root=dataset_config.get("root"), strict_paths=strict)
    from cnom.data.video import VideoDecoder

    source_paths = frame.groupby("source_id")["video_path"].nunique()
    ambiguous = source_paths[source_paths > 1]
    if not ambiguous.empty:
        raise ValueError(f"Each source_id must map to one recording path; ambiguous sources: {ambiguous.index[:5].tolist()}")
    decoder = VideoDecoder(str(dataset_config.get("decoder", "opencv")))
    rows: list[dict] = []
    grouping = ["source_id", "video_path"]
    for (source_id, video_path), group in frame.groupby(grouping, sort=False):
        splits = group["split"].astype(str).unique().tolist()
        if len(splits) != 1:
            raise ValueError(f"Source {source_id} appears in multiple splits: {splits}")
        info = decoder.info(video_path)
        first = group.iloc[0]
        rows.append({
            "sample_id": f"stream:{source_id}", "source_id": str(source_id), "video_path": str(video_path),
            "start_s": 0.0, "end_s": float(info.duration_s), "label": 0, "split": splits[0],
            "camera_id": str(first.get("camera_id", "")), "scene_id": str(first.get("scene_id", "")),
            "event_id": "", "cluster_id": str(first.get("cluster_id", source_id)), "timestamp": first.get("timestamp"),
        })
    result = pd.DataFrame(rows)
    validate_manifest(result, strict_paths=strict)
    return result
