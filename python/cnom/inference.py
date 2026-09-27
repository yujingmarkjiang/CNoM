from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cnom.data.datasets import VideoClipDataset, build_dataloader
from cnom.data.manifest import load_manifest, load_stream_manifest
from cnom.data.video import VideoDecoder, VideoTransform
from cnom.metrics.classification import classification_metrics
from cnom.models.common import ModelOutput
from cnom.models.grounding import CocoGrounder
from cnom.models.preprocessing import resolve_video_preprocessing
from cnom.utils import autocast_dtype, resolve_device, sha256_file, write_table


@torch.inference_mode()
def predict_dataset(model: torch.nn.Module, loader, device: torch.device, precision: str = "fp32") -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval().to(device)
    dtype = autocast_dtype(precision)
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities_all: list[np.ndarray] = []
    latencies: list[float] = []
    for batch in loader:
        inputs = batch["features" if "features" in batch else "video"].to(device, non_blocking=True)
        object_features = batch.get("object_features")
        if object_features is not None:
            object_features = object_features.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type != "cpu"):
            output: ModelOutput = model(inputs, object_features=object_features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        logits = output.logits.detach().cpu().numpy()
        probabilities = torch.from_numpy(logits).softmax(-1).numpy()
        predicted = probabilities.argmax(1)
        labels_batch = batch["label"].detach().cpu().numpy()
        per_item_latency = elapsed / max(1, len(labels_batch))
        segment_scores = output.segment_scores.detach().cpu().numpy() if output.segment_scores is not None else None
        for index, sample_id in enumerate(batch["sample_id"]):
            row: dict[str, Any] = {
                "sample_id": str(sample_id), "label": int(labels_batch[index]), "prediction": int(predicted[index]),
                "confidence": float(probabilities[index].max()), "latency_s": per_item_latency,
            }
            row.update({f"logit_{class_id}": float(value) for class_id, value in enumerate(logits[index])})
            row.update({f"p_{class_id}": float(value) for class_id, value in enumerate(probabilities[index])})
            if segment_scores is not None:
                row["segment_scores"] = segment_scores[index].astype(float).tolist()
            rows.append(row)
        labels.extend(labels_batch.tolist())
        predictions.extend(predicted.tolist())
        probabilities_all.append(probabilities)
        latencies.extend([per_item_latency] * len(labels_batch))
    if not probabilities_all:
        raise ValueError("Evaluation loader produced no batches")
    probabilities_array = np.concatenate(probabilities_all)
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions), probabilities_array, probabilities_array.shape[1])
    metrics.update({
        "median_latency_s": float(np.median(latencies)), "p95_latency_s": float(np.percentile(latencies, 95)),
        "p99_latency_s": float(np.percentile(latencies, 99)),
    })
    return pd.DataFrame(rows), metrics


@torch.inference_mode()
def cache_frame_scores(config: dict[str, Any], model: torch.nn.Module, split: str, output: str | Path) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    frame = load_stream_manifest(dataset_cfg)
    sources = frame.loc[frame["split"].astype(str) == split].sort_values(["source_id", "start_s"]).copy()
    if sources.empty:
        raise ValueError(f"No stream recordings found for split={split}")
    decoder = VideoDecoder(str(dataset_cfg.get("decoder", "opencv")))
    model_cfg = config.get("sentinel", config.get("model", {}))
    preprocessing = resolve_video_preprocessing(model_cfg, dataset_cfg)
    transform = VideoTransform(preprocessing.image_size, train=False, mean=preprocessing.mean, std=preprocessing.std)
    device = resolve_device(str(config.get("device", "auto")))
    precision = str(config.get("precision", "fp32"))
    dtype = autocast_dtype(precision)
    batch_size = int(config.get("runtime", {}).get("score_batch_size", 64))
    analyzed_fps = float(config.get("runtime", {}).get("analyzed_fps", dataset_cfg.get("analyzed_fps", 10)))
    if analyzed_fps <= 0:
        raise ValueError("analyzed_fps must be positive")
    model.eval().to(device)
    rows: list[dict[str, Any]] = []
    for source in sources.itertuples(index=False):
        buffer_frames: list[torch.Tensor] = []
        buffer_times: list[float] = []
        for timestamp_s, image in decoder.iter_frames(source.video_path, float(source.start_s), float(source.end_s), analyzed_fps):
            buffer_frames.append(transform(image.unsqueeze(0))[0])
            buffer_times.append(timestamp_s)
            if len(buffer_frames) == batch_size:
                _score_frame_batch(model, buffer_frames, buffer_times, source, device, dtype, rows)
                buffer_frames, buffer_times = [], []
        if buffer_frames:
            _score_frame_batch(model, buffer_frames, buffer_times, source, device, dtype, rows)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No frames were decoded for stream scoring")
    write_table(result, output)
    return result


def _score_frame_batch(model, frames, times, source, device, dtype, rows) -> None:
    video = torch.stack(frames).unsqueeze(1).to(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type != "cpu"):
        output: ModelOutput = model(video)
    logits = output.frame_logits[:, 0] if output.frame_logits is not None else output.logits
    unsafe = logits[..., 1:] if logits.shape[-1] > 1 else logits
    scores = unsafe.sigmoid().amax(-1).detach().cpu().numpy()
    probabilities = logits.softmax(-1).detach().cpu().numpy()
    for timestamp_s, probability, score in zip(times, probabilities, scores, strict=True):
        rows.append({
            "camera_id": str(getattr(source, "camera_id", "")), "source_id": str(source.source_id),
            "video_path": str(source.video_path), "timestamp_s": float(timestamp_s), "score": float(score),
            "predicted_label": int(probability.argmax()), **{f"p_{index}": float(value) for index, value in enumerate(probability)},
        })


class CandidateTeacherScorer:
    def __init__(self, model: torch.nn.Module, source_map: dict[str, str], config: dict[str, Any]) -> None:
        self.model, self.source_map = model, source_map
        self.device = resolve_device(str(config.get("device", "auto")))
        self.precision = str(config.get("precision", "fp32"))
        dataset_cfg = config["dataset"]
        teacher_cfg = config.get("teacher", config.get("model", {}))
        preprocessing = resolve_video_preprocessing(teacher_cfg, dataset_cfg)
        self.num_frames = preprocessing.num_frames
        self.transform = VideoTransform(preprocessing.image_size, train=False, mean=preprocessing.mean, std=preprocessing.std)
        self.decoder = VideoDecoder(str(dataset_cfg.get("decoder", "opencv")))
        self.batch_size = int(teacher_cfg.get("batch_size", config.get("training", {}).get("batch_size", 8)))
        self.num_classes = int(teacher_cfg.get("num_classes", getattr(model, "num_classes", 0)))
        self.temperature = float(teacher_cfg.get("temperature", config.get("calibration", {}).get("temperature", 1.34)))
        if self.batch_size <= 0 or self.num_classes < 2:
            raise ValueError("Teacher batch_size must be positive and num_classes must be at least 2")
        grounding = dict(config.get("grounding", {}))
        grounding.update(dict(teacher_cfg.get("grounding", {})))
        self.grounder = CocoGrounder(float(grounding.get("score_threshold", 0.35)), tuple(grounding.get("categories", ()))) if grounding.get("enabled") else None
        self.last_total_latency_s = 0.0
        self.last_item_latency_s: list[float] = []
        self.model.eval().to(self.device)
        checkpoint = str(teacher_cfg.get("checkpoint", ""))
        checkpoint_digest = ""
        if checkpoint and Path(checkpoint).is_file():
            checkpoint_digest = sha256_file(checkpoint)
        self.cache_namespace = hashlib.sha256(json.dumps({"teacher": teacher_cfg, "grounding": grounding, "checkpoint": checkpoint_digest, "temperature": self.temperature}, sort_keys=True, default=str).encode()).hexdigest()[:24]

    @torch.inference_mode()
    def __call__(self, candidates: pd.DataFrame) -> np.ndarray:
        if candidates.empty:
            return np.empty((0, self.num_classes), dtype=float)
        all_probabilities: list[np.ndarray] = []
        self.last_total_latency_s = 0.0
        self.last_item_latency_s = []
        videos: list[torch.Tensor] = []
        object_features: list[torch.Tensor] = []
        for candidate in candidates.itertuples(index=False):
            path = self.source_map.get(str(candidate.source_id))
            if path is None:
                raise KeyError(f"No video path for source_id={candidate.source_id}")
            raw = self.decoder.decode(path, float(candidate.start_s), float(candidate.end_s), self.num_frames, "uniform")
            videos.append(self.transform(raw))
            if self.grounder:
                object_features.append(self.grounder.extract(raw.float().div(255.0), self.device))
            if len(videos) == self.batch_size:
                all_probabilities.append(self._score(videos, object_features))
                videos, object_features = [], []
        if videos:
            all_probabilities.append(self._score(videos, object_features))
        probabilities = np.concatenate(all_probabilities)
        if probabilities.shape[1] != self.num_classes or not np.isfinite(probabilities).all():
            raise RuntimeError("Teacher returned invalid class probabilities")
        return probabilities

    def _score(self, videos: list[torch.Tensor], object_features: list[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(videos).to(self.device)
        features = torch.stack(object_features).to(self.device) if self.grounder else None
        dtype = autocast_dtype(self.precision)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None and self.device.type != "cpu"):
            output: ModelOutput = self.model(batch, object_features=features)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.last_total_latency_s += elapsed
        self.last_item_latency_s.extend([elapsed / len(videos)] * len(videos))
        return (output.logits / max(self.temperature, 1e-6)).softmax(-1).detach().cpu().numpy()


def build_clip_loader(config: dict[str, Any], split: str, train: bool = False, model_config: dict[str, Any] | None = None):
    dataset_cfg = config["dataset"]
    model_cfg = model_config or config.get("model") or config.get("teacher") or config.get("sentinel") or {}
    preprocessing = resolve_video_preprocessing(model_cfg, dataset_cfg)
    grounding = dict(config.get("grounding", {}))
    grounding.update(dict(model_cfg.get("grounding", {})))
    dataset = VideoClipDataset(
        dataset_cfg["manifest"], split, root=dataset_cfg.get("root"), num_frames=preprocessing.num_frames,
        image_size=preprocessing.image_size, sampling=str(dataset_cfg.get("sampling", "uniform")),
        decoder=str(dataset_cfg.get("decoder", "opencv")), train=train, strict_paths=bool(dataset_cfg.get("strict_paths", True)),
        include_object_features=bool(grounding.get("enabled", False) and grounding.get("precomputed", True)),
        binary_labels=bool(dataset_cfg.get("binary_labels", False)), object_feature_dim=int(grounding.get("object_dim", 72)),
        mean=preprocessing.mean, std=preprocessing.std, horizontal_flip=0.0, color_jitter=0.0, random_erasing=0.0,
    )
    training_cfg = config.get("training", {})
    return build_dataloader(
        dataset, batch_size=int(training_cfg.get("batch_size", 8)), workers=int(config.get("workers", 4)), shuffle=train,
        class_balance=str(dataset_cfg.get("class_balance", "none")) if train else "none", drop_last=train,
    )
