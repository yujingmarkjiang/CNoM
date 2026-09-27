from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler

from cnom.data.samplers import DistributedEvalSampler, DistributedWeightedSampler

from cnom.data.manifest import load_manifest
from cnom.data.video import VideoDecoder, VideoTransform


class VideoClipDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        split: str,
        root: str | Path | None = None,
        num_frames: int = 32,
        image_size: int = 224,
        sampling: str = "uniform",
        decoder: str = "opencv",
        train: bool = False,
        strict_paths: bool = True,
        include_object_features: bool = False,
        binary_labels: bool = False,
        object_feature_dim: int = 72,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip: float = 0.5,
        color_jitter: float = 0.15,
        random_erasing: float = 0.0,
        secondary_image_size: int | None = None,
        secondary_mean: tuple[float, float, float] | None = None,
        secondary_std: tuple[float, float, float] | None = None,
        secondary_key: str = "teacher_video",
    ) -> None:
        frame = load_manifest(manifest, root=root, strict_paths=strict_paths) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.frame = frame.loc[frame["split"].astype(str) == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples found for split={split}")
        self.decoder = VideoDecoder(decoder)
        self.transform = VideoTransform(image_size=image_size, train=train, mean=mean, std=std, horizontal_flip=horizontal_flip, color_jitter=color_jitter, random_erasing=random_erasing)
        self.secondary_transform = None
        if secondary_image_size is not None:
            self.secondary_transform = VideoTransform(secondary_image_size, train=False, mean=secondary_mean or mean, std=secondary_std or std, horizontal_flip=0.0, color_jitter=0.0, random_erasing=0.0)
        self.secondary_key = secondary_key
        self.num_frames = num_frames
        self.sampling = "random" if train and sampling == "uniform" else sampling
        self.include_object_features = include_object_features
        self.binary_labels = binary_labels
        self.object_feature_dim = object_feature_dim

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        video = self.decoder.decode(row.video_path, float(row.start_s), float(row.end_s), self.num_frames, self.sampling)
        item: dict[str, Any] = {
            "video": self.transform(video),
            "label": torch.tensor(int(int(row.label) != 0) if self.binary_labels else int(row.label), dtype=torch.long),
            "sample_id": str(row.sample_id),
            "source_id": str(row.source_id),
            "camera_id": str(row.get("camera_id", "")),
            "start_s": float(row.start_s),
            "end_s": float(row.end_s),
        }
        if self.secondary_transform is not None:
            item[self.secondary_key] = self.secondary_transform(video)
        if self.include_object_features:
            path = row.get("object_features_path")
            if pd.notna(path) and str(path).strip() and Path(str(path)).is_file():
                item["object_features"] = torch.from_numpy(np.load(str(path))).float()
            else:
                item["object_features"] = torch.zeros(self.object_feature_dim, dtype=torch.float32)
        item["is_hard_negative"] = torch.tensor(bool(row.get("is_hard_negative", False)))
        return item


class FeatureBagDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, manifest: pd.DataFrame | str | Path, split: str, feature_column: str = "feature_path",
        root: str | Path | None = None, num_segments: int | None = None,
    ) -> None:
        frame = load_manifest(manifest, root=root, strict_paths=False) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.frame = frame.loc[frame["split"].astype(str) == split].reset_index(drop=True)
        if num_segments is not None and int(num_segments) <= 0:
            raise ValueError("num_segments must be positive")
        self.feature_column, self.num_segments = feature_column, int(num_segments) if num_segments is not None else None
        if self.frame.empty:
            raise ValueError(f"No feature bags found for split={split}")
        if feature_column not in self.frame.columns:
            raise ValueError(f"Manifest lacks {feature_column}")
        root_path = Path(root).resolve() if root else None
        if root_path is not None:
            self.frame[feature_column] = self.frame[feature_column].map(lambda value: str((root_path / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(value))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(str(row[self.feature_column]))
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.load(path)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"Feature bag must have non-empty shape [segments, features], got {values.shape} for {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Feature bag contains non-finite values: {path}")
        features = torch.from_numpy(values).float()
        if self.num_segments is not None and len(features) != self.num_segments:
            indices = torch.linspace(0, max(0, len(features) - 1), self.num_segments).round().long()
            features = features[indices]
        return {"features": features, "label": torch.tensor(int(row.label)), "sample_id": str(row.sample_id)}


def build_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    workers: int,
    shuffle: bool,
    class_balance: str = "none",
    drop_last: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> DataLoader[Any]:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be non-negative")
    if distributed and (world_size <= 0 or not 0 <= rank < world_size):
        raise ValueError("Distributed rank must satisfy 0 <= rank < world_size")
    sampler = None
    if hasattr(dataset, "frame"):
        frame = getattr(dataset, "frame")
        labels = frame["label"].to_numpy(dtype=int)
        if (labels < 0).any():
            raise ValueError("Class-balanced sampling requires non-negative integer labels")
        weights = np.ones(len(frame), dtype=np.float64)
        if class_balance == "weighted":
            counts = np.bincount(labels)
            weights *= np.array([1 / max(1, counts[label]) for label in labels], dtype=np.float64)
        if "sampling_weight" in frame.columns:
            weights *= frame["sampling_weight"].fillna(1.0).to_numpy(dtype=np.float64)
        if distributed and (class_balance == "weighted" or "sampling_weight" in frame.columns):
            sampler = DistributedWeightedSampler(torch.as_tensor(weights), world_size, rank, seed=seed)
        elif class_balance == "weighted" or "sampling_weight" in frame.columns:
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    if distributed and sampler is None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=drop_last) if shuffle else DistributedEvalSampler(dataset, world_size, rank)
    if sampler is not None:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=bool(drop_last and len(dataset) >= batch_size),
        collate_fn=collate_batch,
    )


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
