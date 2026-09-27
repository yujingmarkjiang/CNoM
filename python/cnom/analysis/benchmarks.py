from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cnom.config import deep_merge, load_config
from cnom.data.manifest import load_manifest, load_stream_manifest
from cnom.inference import CandidateTeacherScorer, build_clip_loader, cache_frame_scores, predict_dataset
from cnom.metrics.events import Event, evaluate_events, events_from_frame
from cnom.models.registry import build_model
from cnom.pipeline.replay import replay_cascade
from cnom.pipeline.routing import Router, RouterConfig, resolve_class_ids
from cnom.pipeline.temporal import TemporalPolicy, generate_candidates
from cnom.utils import read_table, resolve_device, write_table


def run_benchmark_matrix(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(config.get("_config_dir", "."))
    test_split = str(config["dataset"].get("test_split", "test"))
    truth_frame = load_manifest(config["dataset"]["events"], root=config["dataset"].get("root"), strict_paths=False)
    truth_frame = truth_frame.loc[truth_frame["split"].astype(str).eq(test_split)] if "split" in truth_frame.columns else truth_frame
    truth = events_from_frame(truth_frame)
    camera_hours = float(config.get("evaluation", {}).get("camera_hours", _camera_hours(config, test_split)))
    rows, class_rows = [], []
    for system in config["systems"]:
        mode = str(system["mode"])
        if mode == "cloud_only":
            metrics = _cloud_only(config, system, base_dir, truth, camera_hours)
        elif mode == "edge_only":
            metrics = _edge_only(config, system, base_dir, truth, camera_hours, output_dir)
        elif mode == "cascade":
            metrics = _cascade(config, system, truth, camera_hours, output_dir)
        elif mode in {"confidence_offload", "random_offload"}:
            metrics = _clip_offload(config, system, truth, camera_hours, mode)
        elif mode == "rules":
            metrics = _rules(config, system, truth, camera_hours)
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")
        per_class = metrics.pop("per_class")
        rows.append({"system": system["name"], **metrics})
        for row in per_class:
            label = int(row["label"])
            class_rows.append({"system": system["name"], "class": _class_name(config, label), **row})
        (output_dir / f"{system['name']}.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    result = pd.DataFrame(rows)
    write_table(result, output_dir / "system_benchmark.csv")
    write_table(pd.DataFrame(class_rows), output_dir / "class_results.csv")
    return result


def _cloud_only(config, system, base_dir, truth, camera_hours):
    model_cfg = _grounding(load_config(_resolve(base_dir, system["model_config"])), config, "teacher")
    model = build_model(model_cfg, checkpoint=system.get("checkpoint"), strict=False)
    loader = build_clip_loader(deep_merge(config, {"model": dict(model_cfg)}), str(config["dataset"].get("test_split", "test")), train=False)
    predictions, clip_metrics = predict_dataset(model, loader, resolve_device(str(config.get("device", "auto"))), str(config.get("precision", "fp32")))
    manifest = load_manifest(config["dataset"]["manifest"], root=config["dataset"].get("root"), strict_paths=False)
    merged = predictions.merge(manifest[["sample_id", "camera_id", "start_s", "end_s", "source_id"]], on="sample_id", how="left", validate="one_to_one")
    predicted = [Event(str(row.sample_id), str(row.camera_id), float(row.start_s), float(row.end_s), int(row.prediction), float(getattr(row, f"p_{int(row.prediction)}")), str(row.source_id)) for row in merged.itertuples() if int(row.prediction) != 0]
    metrics = evaluate_events(truth, predicted, camera_hours, num_classes=int(model_cfg.get("num_classes", 9)))
    gpu_seconds = float(predictions["latency_s"].sum())
    return _result(config, metrics, camera_hours, 100.0, clip_metrics["median_latency_s"], clip_metrics["p95_latency_s"], np.nan, gpu_seconds, len(predictions), "continuous cloud inference")


def _edge_only(config, system, base_dir, truth, camera_hours, output_dir):
    model_cfg = _grounding(load_config(_resolve(base_dir, system["model_config"])), config, "sentinel")
    model = build_model(model_cfg, checkpoint=system.get("checkpoint"), strict=False)
    if str(model_cfg.get("name")) != "mobilenetv3_sentinel":
        return _clip_edge_only(config, model_cfg, model, truth, camera_hours)
    local = deep_merge(config, {"sentinel": dict(model_cfg)})
    frame_scores = cache_frame_scores(local, model, str(config["dataset"].get("test_split", "test")), output_dir / f"{system['name']}_frame_scores.parquet")
    policy = _temporal_policy(config)
    candidates = generate_candidates(frame_scores, policy)
    predictions: list[Event] = []
    for candidate in candidates.itertuples():
        window = frame_scores.loc[(frame_scores["source_id"].astype(str) == str(candidate.source_id)) & frame_scores["timestamp_s"].between(candidate.start_s, candidate.end_s)]
        label = int(window.loc[window["score"].idxmax(), "predicted_label"]) if not window.empty else 0
        if label:
            predictions.append(Event(candidate.candidate_id, candidate.camera_id, candidate.start_s, candidate.end_s, label, candidate.max_score, candidate.source_id))
    metrics = evaluate_events(truth, predictions, camera_hours, num_classes=int(model_cfg.get("num_classes", 9)))
    edge_fps = _measure_fps(model, int(model_cfg.get("image_size", 224)), 1, config)
    latencies = np.maximum(0.0, candidates["trigger_s"].to_numpy() - candidates["start_s"].to_numpy()) if len(candidates) else np.asarray([np.nan])
    return _result(config, metrics, camera_hours, 0.0, float(np.nanmedian(latencies)), float(np.nanpercentile(latencies, 95)), edge_fps, 0.0, 0, "edge-only frame screening with temporal persistence")


def _clip_edge_only(config, model_cfg, model, truth, camera_hours):
    split = str(config["dataset"].get("test_split", "test"))
    loader = build_clip_loader(deep_merge(config, {"model": dict(model_cfg)}), split, train=False)
    predictions, clip_metrics = predict_dataset(model, loader, resolve_device(str(config.get("device", "auto"))), str(config.get("precision", "fp32")))
    manifest = load_manifest(config["dataset"]["manifest"], root=config["dataset"].get("root"), strict_paths=False)
    columns = ["sample_id", "camera_id", "start_s", "end_s", "source_id"]
    merged = predictions.merge(manifest[columns], on="sample_id", how="left", validate="one_to_one")
    events = [Event(str(row.sample_id), str(row.camera_id), float(row.start_s), float(row.end_s), int(row.prediction), float(row.confidence), str(row.source_id)) for row in merged.itertuples() if int(row.prediction) != 0]
    metrics = evaluate_events(truth, events, camera_hours, num_classes=int(model_cfg.get("num_classes", 9)))
    frames = int(model_cfg.get("num_frames", 16))
    fps = _measure_fps(model, int(model_cfg.get("image_size", 224)), frames, config)
    return _result(config, metrics, camera_hours, 0.0, clip_metrics["median_latency_s"], clip_metrics["p95_latency_s"], fps, 0.0, 0, "edge-only temporal clip classifier")


def _cascade(config, system, truth, camera_hours, output_dir):
    sentinel_cfg = _grounding(dict(config.get("sentinel", {"name": "mobilenetv3_sentinel", "num_classes": 9})), config, "sentinel")
    teacher_cfg = _grounding(dict(config.get("teacher", {"name": "vivit", "num_classes": 9})), config, "teacher")
    sentinel = build_model(sentinel_cfg, checkpoint=system.get("sentinel_checkpoint"), strict=False)
    teacher = build_model(teacher_cfg, checkpoint=system.get("teacher_checkpoint"), strict=False)
    frame_scores = cache_frame_scores(config, sentinel, str(config["dataset"].get("test_split", "test")), output_dir / f"{system['name']}_frame_scores.parquet")
    source_map = {str(key): str(value) for key, value in frame_scores[["source_id", "video_path"]].drop_duplicates().values}
    scorer = CandidateTeacherScorer(teacher, source_map, config)
    selected = config["calibration"]["selected"]
    non_suppressible = resolve_class_ids(config.get("calibration", {}).get("non_suppressible_classes", ()), config.get("labels", ()))
    router = Router(RouterConfig(float(selected["safe_threshold"]), float(selected["unsafe_threshold"]), non_suppressible))
    metrics, routed = replay_cascade(frame_scores, truth, scorer, _temporal_policy(config), router, camera_hours, evaluation={**config.get("evaluation", {}), "num_classes": int(teacher_cfg.get("num_classes", 9))})
    total_s = _stream_duration(config, str(config["dataset"].get("test_split", "test")))
    upload = 100 * float(metrics["uploaded_duration_s"]) / max(total_s, 1e-9)
    network = float(config.get("evaluation", {}).get("fixed_network_latency_s", 0.0))
    teacher_latencies = np.asarray(scorer.last_item_latency_s or [0.0])
    base_latency = routed["end_s"].to_numpy() - routed["trigger_s"].to_numpy() if not routed.empty else np.asarray([np.nan])
    latencies = base_latency + np.resize(teacher_latencies, base_latency.shape) + network
    edge_fps = _measure_fps(sentinel, int(sentinel_cfg.get("image_size", 224)), 1, config)
    return _result(config, metrics, camera_hours, upload, float(np.nanmedian(latencies)), float(np.nanpercentile(latencies, 95)), edge_fps, scorer.last_total_latency_s, len(routed), "semantic edge-cloud cascade")


def _clip_offload(config, system, truth, camera_hours, mode):
    sentinel_cfg = _grounding(dict(config["sentinel"]), config, "sentinel")
    teacher_cfg = _grounding(dict(config["teacher"]), config, "teacher")
    sentinel = build_model(sentinel_cfg, checkpoint=system.get("sentinel_checkpoint", sentinel_cfg.get("checkpoint")), strict=False)
    teacher = build_model(teacher_cfg, checkpoint=system.get("teacher_checkpoint", teacher_cfg.get("checkpoint")), strict=False)
    split = str(config["dataset"].get("test_split", "test"))
    device, precision = resolve_device(str(config.get("device", "auto"))), str(config.get("precision", "fp32"))
    sentinel_loader = build_clip_loader(config, split, train=False, model_config=sentinel_cfg)
    teacher_loader = build_clip_loader(config, split, train=False, model_config=teacher_cfg)
    sentinel_predictions, _ = predict_dataset(sentinel, sentinel_loader, device, precision)
    teacher_predictions, _ = predict_dataset(teacher, teacher_loader, device, precision)
    aligned = _align_offload_predictions(sentinel_predictions, teacher_predictions)
    probability_columns = sorted((column for column in aligned.columns if column.startswith("sentinel_p_")), key=lambda value: int(value.removeprefix("sentinel_p_")))
    teacher_columns = [column.replace("sentinel_", "teacher_", 1) for column in probability_columns]
    probabilities, teacher_probabilities = aligned[probability_columns].to_numpy(float), aligned[teacher_columns].to_numpy(float)
    manifest = load_manifest(config["dataset"]["manifest"], root=config["dataset"].get("root"), strict_paths=False)
    manifest = manifest.loc[manifest["split"].astype(str).eq(split)].copy()
    merged = aligned.merge(manifest, on="sample_id", how="left", validate="one_to_one")
    if merged[["start_s", "end_s"]].isna().any().any():
        raise ValueError("Offload predictions contain sample IDs absent from the evaluation manifest")
    durations = np.clip(merged["end_s"].to_numpy(float) - merged["start_s"].to_numpy(float), 0, None)
    budget = float(system.get("upload_budget", 0.068))
    if not 0 <= budget <= 1:
        raise ValueError("upload_budget must be in [0, 1]")
    if mode == "confidence_offload":
        order = np.argsort(probabilities.max(1), kind="stable")
    else:
        indices = list(range(len(probabilities)))
        random.Random(int(config.get("seed", 2027))).shuffle(indices)
        order = np.asarray(indices, dtype=int)
    selection = _select_duration_budget(order, durations, budget)
    final = probabilities.copy()
    final[selection] = teacher_probabilities[selection]
    merged["prediction"], merged["confidence"] = final.argmax(1), final.max(1)
    predictions = [Event(str(row.sample_id), str(row.camera_id), float(row.start_s), float(row.end_s), int(row.prediction), float(row.confidence), str(row.source_id)) for row in merged.itertuples() if int(row.prediction) != 0]
    metrics = evaluate_events(truth, predictions, camera_hours, num_classes=final.shape[1])
    edge_delay = float(config.get("evaluation", {}).get("offload_edge_decision_latency_s", config["calibration"]["selected"]["window_frames"] / float(config.get("runtime", {}).get("analyzed_fps", 10))))
    latencies = np.full(len(merged), edge_delay, dtype=float)
    teacher_latencies = aligned["teacher_latency_s"].to_numpy(float)
    latencies[selection] += teacher_latencies[selection]
    gpu_seconds = float(teacher_latencies[selection].sum())
    upload_percent = 100 * float(durations[selection].sum()) / max(float(durations.sum()), 1e-9)
    edge_fps = _measure_fps(sentinel, int(sentinel_cfg.get("image_size", 224)), 1, config)
    return _result(config, metrics, camera_hours, upload_percent, float(np.median(latencies)), float(np.percentile(latencies, 95)), edge_fps, gpu_seconds, len(selection), f"{mode} at a duration-matched upload budget")


def _align_offload_predictions(sentinel: pd.DataFrame, teacher: pd.DataFrame) -> pd.DataFrame:
    if sentinel["sample_id"].duplicated().any() or teacher["sample_id"].duplicated().any():
        raise ValueError("Offload prediction tables must contain unique sample IDs")
    sentinel_probability = sorted((column for column in sentinel.columns if column.startswith("p_")), key=lambda value: int(value[2:]))
    teacher_probability = sorted((column for column in teacher.columns if column.startswith("p_")), key=lambda value: int(value[2:]))
    if not sentinel_probability or sentinel_probability != teacher_probability:
        raise ValueError("Sentinel and Teacher predictions must expose the same probability columns")
    left = sentinel[["sample_id", *sentinel_probability]].rename(columns={column: f"sentinel_{column}" for column in sentinel_probability})
    right_columns = ["sample_id", "latency_s", *teacher_probability]
    right = teacher[right_columns].rename(columns={"latency_s": "teacher_latency_s", **{column: f"teacher_{column}" for column in teacher_probability}})
    aligned = left.merge(right, on="sample_id", how="inner", validate="one_to_one")
    if len(aligned) != len(sentinel) or len(aligned) != len(teacher):
        raise ValueError("Sentinel and Teacher prediction sample IDs do not match")
    return aligned


def _select_duration_budget(order: np.ndarray, durations: np.ndarray, budget: float) -> np.ndarray:
    if not len(order) or budget <= 0:
        return np.empty(0, dtype=int)
    target = budget * float(durations.sum())
    selected: list[int] = []
    consumed = 0.0
    for index in np.asarray(order, dtype=int):
        duration = float(durations[index])
        next_consumed = consumed + duration
        if next_consumed <= target or abs(next_consumed - target) < abs(consumed - target):
            selected.append(int(index))
            consumed = next_consumed
        if consumed >= target:
            break
    return np.asarray(selected, dtype=int)


def _rules(config, system, truth, camera_hours):
    signals_path = system.get("signals") or config.get("dataset", {}).get("rule_signals")
    if not signals_path:
        raise ValueError("Rules benchmark requires a signal table with predicted event intervals")
    frame = read_table(signals_path)
    predictions = events_from_frame(frame.loc[frame["label"].astype(int).ne(0)])
    metrics = evaluate_events(truth, predictions, camera_hours)
    return _result(config, metrics, camera_hours, float(system.get("upload_percent", 0)), float(system.get("median_latency_s", 0)), float(system.get("p95_latency_s", 0)), float(system.get("edge_fps", 0)), 0.0, 0, "configured deterministic rules")


def _result(config, metrics, camera_hours, upload_percent, median_latency_s, p95_latency_s, edge_fps, gpu_seconds, cloud_items, notes):
    rate = float(config.get("costing", {}).get("gpu_hour_usd", 0.0))
    monthly_hours = float(config.get("costing", {}).get("camera_hours_per_month", 720.0))
    cloud_cost = gpu_seconds / max(camera_hours, 1e-9) * monthly_hours / 3600 * rate
    return {
        "edge_fps": edge_fps, "event_tp": metrics["tp"], "event_fn": metrics["fn"], "event_fp": metrics["fp"], "camera_hours": camera_hours,
        "reported_recall": metrics["recall"], "recall_ci_low": metrics["recall_ci"][0], "recall_ci_high": metrics["recall_ci"][1],
        "reported_precision": metrics["precision"], "precision_ci_low": metrics["precision_ci"][0], "precision_ci_high": metrics["precision_ci"][1],
        "reported_macro_f1": metrics["macro_f1"], "false_alerts_per_camera_hour": metrics["false_alerts_per_camera_hour"],
        "upload_percent": upload_percent, "median_latency_s": median_latency_s, "p95_latency_s": p95_latency_s,
        "cloud_items": cloud_items, "cloud_gpu_seconds": gpu_seconds, "cloud_cost_per_camera_month": cloud_cost, "notes": notes,
        "per_class": metrics["per_class"],
    }


def _temporal_policy(config):
    selected, runtime = config["calibration"]["selected"], config.get("runtime", {})
    return TemporalPolicy(float(selected["frame_threshold"]), int(selected["window_frames"]), float(selected["persistence"]), float(selected["merge_seconds"]), float(selected["cooldown_seconds"]), float(runtime.get("pre_roll_s", 2)), float(runtime.get("post_roll_s", 4.8)), float(runtime.get("max_clip_s", 12)))


def _measure_fps(model, size, frames, config):
    device = resolve_device(str(config.get("device", "auto")))
    model.eval().to(device)
    batch = torch.randn(1, frames, 3, size, size, device=device)
    dtype = torch.float16 if str(config.get("precision")) == "fp16" and device.type == "cuda" else None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        for _ in range(5):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started, repetitions = time.perf_counter(), int(config.get("evaluation", {}).get("fps_repetitions", 50))
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        for _ in range(repetitions):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return repetitions * frames / (time.perf_counter() - started)


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _stream_duration(config, split):
    frame = load_stream_manifest(config["dataset"], strict_paths=False)
    frame = frame.loc[frame["split"].astype(str).eq(split)]
    return float((frame["end_s"] - frame["start_s"]).sum())


def _camera_hours(config, split):
    return _stream_duration(config, split) / 3600


def _grounding(model_cfg, config, role):
    grounding = dict(config.get("grounding", {}))
    if grounding.get("enabled") and role in set(grounding.get("apply_to", ("teacher",))):
        model_cfg = dict(model_cfg)
        model_cfg["grounding"] = {**grounding, "enabled": True}
    return model_cfg


def _class_name(config: dict[str, Any], label: int) -> str:
    labels = list(config.get("labels", ()))
    return labels[label] if 0 <= label < len(labels) else str(label)
