from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cnom.utils import read_table, write_table


_DEFAULT_COMPONENTS = {
    "capture_decode": ("frame_available_ts", "tensor_ready_ts"),
    "sentinel_inference": ("sentinel_started_ts", "sentinel_completed_ts"),
    "persistence_clip_assembly": ("event_onset_ts", "clip_materialized_ts"),
    "encoding": ("encoding_started_ts", "encoding_completed_ts"),
    "upload": ("upload_started_ts", "upload_completed_ts"),
    "ingress_validation": ("upload_completed_ts", "ingress_accepted_ts"),
    "cloud_queue": ("ingress_accepted_ts", "teacher_started_ts"),
    "teacher_inference": ("teacher_started_ts", "teacher_completed_ts"),
    "dashboard_delivery": ("teacher_completed_ts", "dashboard_visible_ts"),
    "end_to_end_onset_to_alert": ("event_onset_ts", "dashboard_visible_ts"),
    "onset_to_acknowledgement": ("event_onset_ts", "acknowledged_ts"),
}


def analyze_latency(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    section = config["latency"]
    frame = read_table(section["traces"]).copy()
    components = section.get("components", _DEFAULT_COMPONENTS)
    output_dir = Path(config.get("output_dir", "runs/latency"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    event = frame[[column for column in ("candidate_id", "camera_id", "source_id") if column in frame.columns]].copy()
    for name, pair in components.items():
        start, end = pair
        if start not in frame.columns or end not in frame.columns:
            if bool(section.get("strict_columns", True)):
                raise ValueError(f"Latency component {name} requires {start} and {end}")
            continue
        start_ts, end_ts = pd.to_datetime(frame[start], utc=True, errors="coerce"), pd.to_datetime(frame[end], utc=True, errors="coerce")
        values = (end_ts - start_ts).dt.total_seconds() * 1000
        invalid = values < 0
        if invalid.any():
            raise ValueError(f"Latency component {name} contains {int(invalid.sum())} negative durations")
        event[f"{name}_ms"] = values
        valid = values.dropna().to_numpy(float)
        rows.append(_summary(name, valid))
    summary = pd.DataFrame(rows)
    write_table(event, output_dir / "latency_events.parquet")
    write_table(summary, output_dir / "latency_components.csv")
    if section.get("clock_offsets"):
        offsets = read_table(section["clock_offsets"])
        column = str(section.get("clock_offset_column", "absolute_offset_ms"))
        values = pd.to_numeric(offsets[column], errors="coerce").dropna().to_numpy(float)
        clock = pd.DataFrame([_summary("absolute_clock_offset", values)])
        write_table(clock, output_dir / "clock_offsets.csv")
    else:
        clock = pd.DataFrame()
    return {"latency_events": event, "latency_components": summary, "clock_offsets": clock}


def _summary(name: str, values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"component": name, "n": 0, "mean_ms": np.nan, "median_ms": np.nan, "p95_ms": np.nan, "p99_ms": np.nan, "max_ms": np.nan}
    return {
        "component": name, "n": len(values), "mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)), "max_ms": float(np.max(values)),
    }
