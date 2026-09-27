from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from cnom.metrics.events import Event, evaluate_events
from cnom.metrics.statistics import wilson_interval
from cnom.pipeline.routing import RouteDecision, Router, RouterConfig, resolve_class_ids
from cnom.pipeline.temporal import TemporalPolicy, generate_candidates
from cnom.utils import write_table


class CandidateProbabilityCache:
    def __init__(self, path: str | Path | None = None, namespace: str = "default") -> None:
        self.path = Path(path) if path else None
        self.namespace = str(namespace).strip() or "default"
        self.values: dict[str, list[float]] = {}
        if self.path and self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.values = {str(key): list(map(float, value)) for key, value in payload.items()}

    def get_or_score(self, candidates: pd.DataFrame, scorer: Callable[[pd.DataFrame], np.ndarray]) -> np.ndarray:
        keys = candidates.apply(self._key, axis=1).tolist()
        missing_rows = [index for index, key in enumerate(keys) if key not in self.values]
        if missing_rows:
            probabilities = np.asarray(scorer(candidates.iloc[missing_rows]), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[0] != len(missing_rows):
                raise ValueError("Candidate scorer must return a two-dimensional [candidates, classes] array")
            if probabilities.shape[1] < 2 or not np.isfinite(probabilities).all():
                raise ValueError("Candidate scorer returned invalid class probabilities")
            for row_index, probability in zip(missing_rows, probabilities, strict=True):
                self.values[keys[row_index]] = probability.astype(float).tolist()
            self.flush()
        result = np.asarray([self.values[key] for key in keys], dtype=float)
        if result.ndim != 2 or len({len(self.values[key]) for key in keys}) != 1:
            raise ValueError("Cached candidate probabilities have inconsistent class dimensions")
        return result

    def _key(self, row: pd.Series) -> str:
        return f"{self.namespace}|{row['source_id']}|{float(row['start_s']):.6f}|{float(row['end_s']):.6f}"

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.values), encoding="utf-8")


def calibrate_policy(
    frame_scores: pd.DataFrame,
    truth: list[Event],
    teacher_scorer: Callable[[pd.DataFrame], np.ndarray],
    config: dict[str, Any],
    camera_hours: float,
    stream_duration_s: float,
    output_dir: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = config["calibration"]
    runtime = config.get("runtime", {})
    temporal_combinations = itertools.product(
        calibration["frame_thresholds"], calibration["window_frames"], calibration["persistence"],
        calibration["merge_seconds"], calibration["cooldown_seconds"],
    )
    routing_combinations = list(itertools.product(calibration["safe_thresholds"], calibration["unsafe_thresholds"]))
    cache = CandidateProbabilityCache(output_dir / "candidate_probabilities.json", namespace=getattr(teacher_scorer, "cache_namespace", "default"))
    rows: list[dict[str, Any]] = []
    non_suppressible = resolve_class_ids(calibration.get("non_suppressible_classes", ()), config.get("labels", ()))
    for frame_threshold, window_frames, persistence, merge_seconds, cooldown_seconds in temporal_combinations:
        temporal = TemporalPolicy(
            frame_threshold=float(frame_threshold), window_frames=int(window_frames), persistence=float(persistence),
            merge_seconds=float(merge_seconds), cooldown_seconds=float(cooldown_seconds),
            pre_roll_s=float(runtime.get("pre_roll_s", 1.6)), post_roll_s=float(runtime.get("post_roll_s", 1.6)),
            max_clip_s=float(runtime.get("max_clip_s", 12.0)),
        )
        candidates = generate_candidates(frame_scores, temporal)
        if candidates.empty:
            for safe_threshold, unsafe_threshold in routing_combinations:
                rows.append(_empty_row(temporal, safe_threshold, unsafe_threshold))
            continue
        probabilities = cache.get_or_score(candidates, teacher_scorer)
        clip_time = float((candidates["end_s"] - candidates["start_s"]).sum())
        upload_percent = 100 * clip_time / max(stream_duration_s, 1e-9)
        evidence_wait = np.maximum(0, candidates["end_s"].to_numpy() - candidates["trigger_s"].to_numpy())
        teacher_latency = float(calibration.get("teacher_latency_s", 0.35))
        p95_latency = float(np.percentile(evidence_wait + teacher_latency, 95))
        for safe_threshold, unsafe_threshold in routing_combinations:
            router = Router(RouterConfig(float(safe_threshold), float(unsafe_threshold), non_suppressible))
            predicted: list[Event] = []
            manual_review = 0
            for candidate, probability in zip(candidates.itertuples(), probabilities, strict=True):
                routed = router.route(probability)
                if routed.decision != RouteDecision.SUPPRESSED_SAFE:
                    predicted.append(Event(candidate.candidate_id, candidate.camera_id, candidate.start_s, candidate.end_s, routed.label, routed.confidence, candidate.source_id))
                    if routed.decision == RouteDecision.MANUAL_REVIEW:
                        manual_review += 1
            metrics = evaluate_events(
                truth, predicted, camera_hours, tiou_threshold=float(config.get("evaluation", {}).get("tiou_threshold", 0.3)),
                onset_tolerance_s=float(config.get("evaluation", {}).get("onset_tolerance_s", 2.0)),
                class_aware=bool(config.get("evaluation", {}).get("class_aware_matching", True)),
            )
            lower = wilson_interval(int(metrics["tp"]), int(metrics["tp"] + metrics["fn"]))[0]
            shifts = max(1.0, float(calibration.get("validation_shifts", camera_hours / 8)))
            row = {
                **asdict(temporal), "safe_threshold": float(safe_threshold), "unsafe_threshold": float(unsafe_threshold),
                "tp": metrics["tp"], "fn": metrics["fn"], "fp": metrics["fp"], "recall": metrics["recall"],
                "recall_lower_bound": lower, "precision": metrics["precision"], "macro_f1": metrics["macro_f1"],
                "false_alerts_per_camera_hour": metrics["false_alerts_per_camera_hour"], "upload_percent": upload_percent,
                "p95_latency_s": p95_latency, "manual_review_per_shift": manual_review / shifts, "candidate_events": len(candidates),
            }
            row["feasible"] = lower >= float(calibration["target_recall_lower_bound"]) and p95_latency <= float(calibration["max_p95_latency_s"]) and row["manual_review_per_shift"] <= float(calibration["max_review_items_per_shift"])
            row["objective"] = upload_percent + float(calibration.get("false_alert_weight", 4.0)) * row["false_alerts_per_camera_hour"] + float(calibration.get("latency_weight", 0.15)) * p95_latency
            rows.append(row)
    frontier = pd.DataFrame(rows)
    feasible = frontier.loc[frontier["feasible"]].sort_values(["objective", "recall"], ascending=[True, False])
    if feasible.empty:
        write_table(frontier, output_dir / "calibration_frontier.parquet")
        raise RuntimeError("No calibration policy satisfies the configured safety and deployment constraints")
    selected = feasible.iloc[0].to_dict()
    write_table(frontier, output_dir / "calibration_frontier.parquet")
    (output_dir / "selected_policy.json").write_text(json.dumps(selected, indent=2, default=_json_default), encoding="utf-8")
    return selected, frontier


def _empty_row(temporal: TemporalPolicy, safe_threshold: float, unsafe_threshold: float) -> dict[str, Any]:
    return {**asdict(temporal), "safe_threshold": safe_threshold, "unsafe_threshold": unsafe_threshold, "tp": 0, "fn": 0, "fp": 0, "recall": 0.0, "recall_lower_bound": 0.0, "precision": 0.0, "macro_f1": 0.0, "false_alerts_per_camera_hour": 0.0, "upload_percent": 0.0, "p95_latency_s": float("inf"), "manual_review_per_shift": 0.0, "candidate_events": 0, "feasible": False, "objective": float("inf")}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)
