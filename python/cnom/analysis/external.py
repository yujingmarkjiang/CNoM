from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from cnom.config import deep_merge, load_config
from cnom.data.adapters import prepare_manifest
from cnom.data.manifest import load_manifest
from cnom.data.video import VideoDecoder
from cnom.inference import build_clip_loader, predict_dataset
from cnom.models.registry import build_model
from cnom.runner import evaluate_experiment, train_experiment
from cnom.utils import read_table, resolve_device, write_table


def run_external_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(config.get("_config_dir", "."))
    rows: list[dict[str, Any]] = []
    for benchmark in config["benchmarks"]:
        dataset_config = load_config(_resolve(base_dir, benchmark["config"]))
        manifest_path = Path(dataset_config.get("output", output_dir / f"{benchmark['dataset']}.parquet"))
        if not manifest_path.is_file():
            prepare_manifest(dataset_config, output=str(manifest_path))
        for protocol in benchmark["protocols"]:
            if protocol in {"full_finetune", "linear_probe"}:
                result = _train_and_evaluate(config, benchmark, protocol, manifest_path, output_dir, base_dir)
            elif protocol == "zero_shot":
                result = _zero_shot_transfer(config, benchmark, manifest_path, base_dir)
            elif protocol == "anomaly":
                result = _anomaly_metric(config, benchmark, manifest_path, output_dir, base_dir)
            elif protocol == "streaming":
                result = _streaming_metric(config, benchmark, manifest_path, base_dir)
            elif protocol == "vlm_zero_shot":
                result = _prediction_file_metric(benchmark, base_dir)
            else:
                raise ValueError(f"Unknown external protocol: {protocol}")
            rows.append({"dataset": benchmark["dataset"], "protocol": protocol, "primary_metric": benchmark["metric"], **result})
    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "external_benchmarks.csv")
    return frame


def _train_and_evaluate(config, benchmark, protocol, manifest_path, output_dir, base_dir):
    manifest = load_manifest(manifest_path, strict_paths=False)
    num_classes = int(manifest["label"].max()) + 1
    run_dir = output_dir / str(benchmark["dataset"]) / protocol
    model_cfg = _benchmark_model(config, benchmark, base_dir)
    model_cfg.update({"num_classes": num_classes, "checkpoint": benchmark.get("initial_checkpoint", config.get("checkpoint"))})
    local = deep_merge(config, {
        "output_dir": str(run_dir), "stage": "teacher", "dataset": {"manifest": str(manifest_path), "events": str(manifest_path)},
        "model": model_cfg, "training": {"epochs": int(benchmark.get("epochs", 20))},
    })
    if protocol == "linear_probe":
        local["training"]["freeze_backbone"] = True
        local["training"]["epochs"] = int(benchmark.get("linear_probe_epochs", 15))
    train_experiment(local)
    predictions, metrics = evaluate_experiment(local, str(local["dataset"].get("test_split", "test")))
    value, low, high = _classification_value_ci(predictions, str(benchmark["metric"]), config)
    return {"value": value, "ci_low": low, "ci_high": high, "latency_ms": 1000 * metrics["median_latency_s"], "note": "trained on the benchmark training split"}


def _zero_shot_transfer(config, benchmark, manifest_path, base_dir):
    manifest = load_manifest(manifest_path, strict_paths=False)
    test_split = str(benchmark.get("test_split", "test"))
    if set(manifest.loc[manifest["split"].astype(str).eq(test_split), "label"].astype(int).unique()).difference({0, 1}):
        raise ValueError(f"Zero-shot unsafe transfer currently requires binary labels for {benchmark['dataset']}")
    model_cfg = _benchmark_model(config, benchmark, base_dir)
    model_cfg["num_classes"] = int(benchmark.get("source_num_classes", len(config.get("labels", ())) or 9))
    checkpoint = benchmark.get("checkpoint", config.get("checkpoint", model_cfg.get("checkpoint")))
    model = build_model(model_cfg, checkpoint=checkpoint, strict=False)
    local = deep_merge(config, {"dataset": {"manifest": str(manifest_path), "binary_labels": True}, "model": model_cfg})
    predictions, metrics = predict_dataset(
        model, build_clip_loader(local, test_split, train=False, model_config=model_cfg),
        resolve_device(str(config.get("device", "auto"))), str(config.get("precision", "fp32")),
    )
    probability_columns = sorted((column for column in predictions.columns if column.startswith("p_")), key=lambda value: int(value[2:]))
    predictions["score"] = 1.0 - predictions["p_0"]
    predictions["prediction"] = (predictions["score"] >= float(benchmark.get("threshold", 0.5))).astype(int)
    value, low, high = _classification_value_ci(predictions, str(benchmark["metric"]), config)
    return {"value": value, "ci_low": low, "ci_high": high, "latency_ms": 1000 * metrics["median_latency_s"], "note": f"direct unsafe-probability transfer using {len(probability_columns)} source classes"}


def _anomaly_metric(config, benchmark, manifest_path, output_dir, base_dir):
    run_dir = output_dir / str(benchmark["dataset"]) / "anomaly"
    model_cfg = _benchmark_model(config, benchmark, base_dir)
    model_cfg.update({"num_classes": 2, "checkpoint": benchmark.get("checkpoint", model_cfg.get("checkpoint"))})
    local = deep_merge(config, {
        "output_dir": str(run_dir), "dataset": {"manifest": str(manifest_path), "feature_column": benchmark.get("feature_column", "feature_path")},
        "model": model_cfg,
    })
    predictions, metrics = evaluate_experiment(local, str(benchmark.get("test_split", "test")))
    metric = str(benchmark["metric"])
    if metric.startswith("frame_"):
        labels, scores = _frame_anomaly_arrays(predictions, load_manifest(manifest_path, strict_paths=False), str(benchmark.get("test_split", "test")))
        function = average_precision_score if metric == "frame_average_precision" else roc_auc_score
        note = f"frame-level anomaly score over {len(labels)} frames"
    else:
        score = predictions.get("p_1")
        if score is None:
            raise ValueError("Anomaly evaluation requires p_1 scores")
        labels, scores = predictions["label"].to_numpy(int), score.to_numpy(float)
        function = average_precision_score if metric == "average_precision" else roc_auc_score
        note = "video-level anomaly score"
    value, low, high = _bootstrap_metric(labels, scores, function, config)
    return {"value": value, "ci_low": low, "ci_high": high, "latency_ms": 1000 * metrics["median_latency_s"], "note": note}


def _streaming_metric(config, benchmark, manifest_path, base_dir):
    frame = load_manifest(manifest_path, strict_paths=False)
    predictions_path = benchmark.get("predictions")
    if not predictions_path:
        raise ValueError(f"Streaming benchmark {benchmark['dataset']} requires a predictions path")
    predictions = read_table(_resolve(base_dir, predictions_path))
    required = {"sample_id", "warned", "warning_time_s", "latency_s", "risk_onset_s"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Streaming predictions lack columns: {sorted(missing)}")
    merged = frame.merge(predictions, on="sample_id", validate="one_to_one")
    score = _warning_score(merged)
    safe_fp, recall, lead = _warning_components(merged)
    rng = np.random.default_rng(int(config.get("seed", 2027)))
    repetitions = int(config.get("evaluation", {}).get("bootstrap_repetitions", 2000))
    values: list[float] = []
    for _ in range(repetitions):
        sample = merged.iloc[rng.integers(0, len(merged), len(merged))]
        try:
            values.append(_warning_score(sample))
        except ValueError:
            continue
    if not values:
        raise ValueError("Streaming bootstrap samples did not contain both safe and risky examples")
    alpha = (1 - float(config.get("evaluation", {}).get("confidence_level", 0.95))) / 2
    return {
        "value": score, "ci_low": float(np.quantile(values, alpha)), "ci_high": float(np.quantile(values, 1 - alpha)),
        "latency_ms": float(1000 * predictions["latency_s"].median()),
        "note": f"recall={recall:.4f}; safe_false_warning={safe_fp:.4f}; lead_s={lead:.4f}",
    }


def _prediction_file_metric(benchmark, base_dir):
    path = benchmark.get("predictions")
    if not path:
        raise ValueError(f"VLM benchmark {benchmark['dataset']} requires a predictions path generated by vlm-diagnostic")
    frame = read_table(_resolve(base_dir, path))
    required = {"label", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"VLM predictions lack columns: {sorted(missing)}")
    metric = str(benchmark["metric"])
    value = _classification_metric(frame["label"].to_numpy(int), frame["prediction"].to_numpy(int), metric)
    low, high = _bootstrap_classification(
        frame["label"].to_numpy(int), frame["prediction"].to_numpy(int), metric,
        int(benchmark.get("bootstrap_repetitions", 2000)), int(benchmark.get("seed", 2027)), float(benchmark.get("confidence_level", 0.95)),
    )
    latency = float(frame["latency_ms"].median()) if "latency_ms" in frame.columns else float(frame["latency_s"].median() * 1000)
    return {"value": value, "ci_low": low, "ci_high": high, "latency_ms": latency, "note": "versioned VLM diagnostic predictions"}


def _classification_value_ci(frame: pd.DataFrame, metric: str, config: dict[str, Any]) -> tuple[float, float, float]:
    labels, predictions = frame["label"].to_numpy(int), frame["prediction"].to_numpy(int)
    value = _classification_metric(labels, predictions, metric)
    repetitions = int(config.get("evaluation", {}).get("bootstrap_repetitions", 2000))
    low, high = _bootstrap_classification(
        labels, predictions, metric, repetitions, int(config.get("seed", 2027)),
        float(config.get("evaluation", {}).get("confidence_level", 0.95)),
    )
    return value, low, high


def _classification_metric(labels: np.ndarray, predictions: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(labels, predictions))
    if metric in {"macro_f1", "f1"}:
        return float(f1_score(labels, predictions, average="macro" if metric == "macro_f1" else "binary", zero_division=0))
    raise ValueError(f"Unsupported classification metric: {metric}")


def _bootstrap_classification(labels, predictions, metric, repetitions, seed, confidence=0.95):
    rng, values = np.random.default_rng(seed), np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.integers(0, len(labels), len(labels))
        values[index] = _classification_metric(labels[sample], predictions[sample], metric)
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def _bootstrap_metric(labels: np.ndarray, scores: np.ndarray, function: Callable, config: dict[str, Any]):
    value = float(function(labels, scores))
    repetitions = int(config.get("evaluation", {}).get("bootstrap_repetitions", 2000))
    rng, values = np.random.default_rng(int(config.get("seed", 2027))), []
    for _ in range(repetitions):
        sample = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[sample])) < 2:
            continue
        values.append(float(function(labels[sample], scores[sample])))
    if not values:
        raise ValueError("Bootstrap samples did not contain both anomaly classes")
    alpha = (1 - float(config.get("evaluation", {}).get("confidence_level", 0.95))) / 2
    return value, float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))



def _frame_anomaly_arrays(predictions: pd.DataFrame, manifest: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    required = {"sample_id", "segment_scores"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Frame-level anomaly predictions lack columns: {sorted(missing)}")
    selected = manifest.loc[manifest["split"].astype(str).eq(split)].copy()
    merged = selected.merge(predictions[["sample_id", "segment_scores"]], on="sample_id", validate="one_to_one")
    if len(merged) != len(selected):
        raise ValueError("Frame-level anomaly predictions do not cover every selected manifest sample")
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    decoder = VideoDecoder()
    for row in merged.itertuples(index=False):
        segment_scores = np.asarray(row.segment_scores, dtype=float).reshape(-1)
        if segment_scores.size == 0:
            raise ValueError(f"Empty segment scores for {row.sample_id}")
        labels = _frame_labels_for_row(row, decoder)
        positions = np.linspace(0, max(0, len(labels) - 1), segment_scores.size)
        scores = np.interp(np.arange(len(labels)), positions, segment_scores)
        labels_all.append(labels.astype(int))
        scores_all.append(scores.astype(float))
    if not labels_all:
        raise ValueError("Frame-level anomaly evaluation found no samples")
    labels, scores = np.concatenate(labels_all), np.concatenate(scores_all)
    if len(np.unique(labels)) < 2:
        raise ValueError("Frame-level anomaly metric requires both normal and anomalous frames")
    return labels, scores


def _frame_labels_for_row(row, decoder: VideoDecoder) -> np.ndarray:
    mask_path = getattr(row, "frame_labels_path", None)
    if mask_path and str(mask_path) not in {"nan", "None"} and Path(str(mask_path)).is_file():
        path = Path(str(mask_path))
        if path.suffix.lower() == ".npy":
            return np.load(path).reshape(-1).astype(int)
        values = np.fromstring(path.read_text(encoding="utf-8", errors="ignore").replace(",", " "), sep=" ")
        if values.size:
            return values.reshape(-1).astype(int)
    info = decoder.info(row.video_path)
    labels = np.zeros(info.frames, dtype=int)
    annotations = getattr(row, "temporal_annotations", None)
    intervals = []
    if annotations and str(annotations) not in {"nan", "None", "[]"}:
        import json
        intervals = json.loads(str(annotations)) if isinstance(annotations, str) else annotations
    for start, end in intervals:
        labels[max(0, int(start)):min(info.frames, int(end))] = 1
    if int(row.label) != 0 and not intervals:
        raise ValueError(f"Frame labels are unavailable for anomalous sample {row.sample_id}")
    return labels

def _warning_score(frame: pd.DataFrame) -> float:
    safe_fp, recall, lead = _warning_components(frame)
    return float(recall * np.exp(-safe_fp) * min(1.0, max(0.0, lead / 3.0)))


def _warning_components(frame: pd.DataFrame) -> tuple[float, float, float]:
    safe = frame.loc[frame["label"].astype(int).eq(0)]
    risky = frame.loc[frame["label"].astype(int).ne(0)]
    if safe.empty or risky.empty:
        raise ValueError("Streaming warning score requires both safe and risky samples")
    safe_fp = float(safe["warned"].astype(bool).mean())
    recall = float(risky["warned"].astype(bool).mean())
    warned = risky.loc[risky["warned"].astype(bool)]
    lead = float((warned["risk_onset_s"] - warned["warning_time_s"]).clip(lower=0).mean()) if not warned.empty else 0.0
    return safe_fp, recall, lead


def _benchmark_model(config, benchmark, base_dir):
    if benchmark.get("model_config"):
        return dict(load_config(_resolve(base_dir, benchmark["model_config"])))
    return dict(config.get("teacher", config.get("model", {})))


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()
