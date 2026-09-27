from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cnom.utils import read_table, write_table


@dataclass(frozen=True)
class QueueResult:
    cameras: int
    workers: int
    arrival_rate_per_s: float
    utilization: float
    median_queue_s: float
    p95_queue_s: float
    p99_queue_s: float
    expired_percent: float
    completed: int


def analyze_resource_usage(config: dict[str, Any]) -> pd.DataFrame:
    resource = config["resource"]
    candidates = read_table(resource["candidates"])
    streams = read_table(resource["streams"])
    total_duration = float((streams["end_s"] - streams["start_s"]).sum())
    uploaded_duration = float((candidates["end_s"] - candidates["start_s"]).sum())
    camera_hours = total_duration / 3600
    upload_percent = 100 * uploaded_duration / max(total_duration, 1e-9)
    uploaded_bytes = float(candidates[resource["bytes_column"]].sum()) if resource.get("bytes_column") in candidates.columns else np.nan
    continuous_bitrate_mbps = float(resource.get("continuous_bitrate_mbps", 1.2))
    continuous_bytes = total_duration * continuous_bitrate_mbps * 1e6 / 8
    teacher_windows = int(candidates[resource["windows_column"]].sum()) if resource.get("windows_column") in candidates.columns else len(candidates)
    service_s = float(resource.get("teacher_mean_service_s", 0.35))
    gpu_hours = teacher_windows * service_s / 3600
    gpu_price = float(resource.get("gpu_price_per_hour", 1.25))
    storage_price = float(resource.get("storage_price_per_gb", 0.023))
    egress_price = float(resource.get("egress_price_per_gb", 0.0))
    upload_gb = uploaded_bytes / 1e9 if np.isfinite(uploaded_bytes) else continuous_bytes * upload_percent / 100 / 1e9
    cloud_cost = gpu_hours * gpu_price + upload_gb * (storage_price + egress_price)
    continuous_gpu_hours = total_duration * float(resource.get("continuous_windows_per_s", 0.5)) * service_s / 3600
    continuous_cost = continuous_gpu_hours * gpu_price + continuous_bytes / 1e9 * (storage_price + egress_price)
    row = {
        "camera_hours": camera_hours, "candidate_events": len(candidates), "candidate_events_per_camera_hour": len(candidates) / max(camera_hours, 1e-9),
        "uploaded_duration_s": uploaded_duration, "upload_percent": upload_percent, "uploaded_gb": upload_gb,
        "teacher_windows": teacher_windows, "teacher_gpu_hours": gpu_hours, "estimated_cloud_cost": cloud_cost,
        "continuous_cloud_gpu_hours": continuous_gpu_hours, "continuous_cloud_cost": continuous_cost,
        "bandwidth_reduction_percent": 100 * (1 - upload_gb / max(continuous_bytes / 1e9, 1e-9)),
        "cloud_cost_reduction_percent": 100 * (1 - cloud_cost / max(continuous_cost, 1e-9)),
    }
    result = pd.DataFrame([row])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(result, output_dir / "resource_usage.csv")
    return result


def run_capacity_study(config: dict[str, Any]) -> pd.DataFrame:
    capacity = config["capacity"]
    rows: list[dict[str, Any]] = []
    for cameras in capacity["camera_counts"]:
        for workers in capacity["worker_counts"]:
            for seed in capacity.get("seeds", [2027, 2028, 2029]):
                result = simulate_queue(int(cameras), int(workers), capacity, int(seed))
                rows.append({**result.__dict__, "seed": int(seed)})
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["cameras", "workers"], as_index=False).agg(
        arrival_rate_per_s=("arrival_rate_per_s", "mean"), utilization=("utilization", "mean"), median_queue_s=("median_queue_s", "mean"),
        p95_queue_s=("p95_queue_s", "mean"), p99_queue_s=("p99_queue_s", "mean"), expired_percent=("expired_percent", "mean"), completed=("completed", "mean"),
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(raw, output_dir / "capacity_runs.csv")
    write_table(summary, output_dir / "capacity_summary.csv")
    return summary


def simulate_queue(cameras: int, workers: int, config: dict[str, Any], seed: int) -> QueueResult:
    if cameras <= 0 or workers <= 0:
        raise ValueError("cameras and workers must be positive")
    rng = np.random.default_rng(seed)
    duration_s = float(config.get("duration_hours", 24)) * 3600
    rate_per_camera_hour = float(config.get("candidate_events_per_camera_hour", 4.5))
    if duration_s <= 0 or rate_per_camera_hour < 0:
        raise ValueError("duration must be positive and event rate must be non-negative")
    arrival_rate = cameras * rate_per_camera_hour / 3600
    count = rng.poisson(arrival_rate * duration_s)
    arrivals = np.sort(rng.uniform(0, duration_s, count))
    burst_probability = float(config.get("burst_probability", 0.02))
    burst_size = int(config.get("burst_size", 8))
    if burst_size <= 0 or not 0 <= burst_probability <= 1:
        raise ValueError("burst_size must be positive and burst_probability must be within [0, 1]")
    burst_mask = rng.random(count) < burst_probability
    arrivals = np.sort(np.concatenate([arrivals, np.repeat(arrivals[burst_mask], burst_size - 1)]))
    median = float(config.get("service_median_s", 0.35))
    p95 = float(config.get("service_p95_s", 0.62))
    if median <= 0 or p95 < median:
        raise ValueError("Service times must be positive with p95 at least the median")
    sigma = max(1e-6, (np.log(p95) - np.log(median)) / 1.6448536269514722)
    service = rng.lognormal(np.log(median), sigma, len(arrivals))
    available = [0.0] * workers
    heapq.heapify(available)
    waits: list[float] = []
    expired = 0
    max_queue_s = float(config.get("max_queue_s", 30))
    outage_start = float(config.get("outage_start_hour", -1)) * 3600
    outage_duration = float(config.get("outage_duration_s", 0))
    for arrival, duration in zip(arrivals, service, strict=True):
        worker_available = heapq.heappop(available)
        start = max(arrival, worker_available)
        if outage_start >= 0 and outage_start <= start < outage_start + outage_duration:
            start = outage_start + outage_duration
        wait = start - arrival
        if wait > max_queue_s:
            expired += 1
            heapq.heappush(available, worker_available)
            continue
        waits.append(wait)
        heapq.heappush(available, start + duration)
    values = np.asarray(waits) if waits else np.asarray([0.0])
    busy = float(service.sum())
    utilization = busy / max(duration_s * workers, 1e-9)
    return QueueResult(cameras, workers, arrival_rate, utilization, float(np.median(values)), float(np.percentile(values, 95)), float(np.percentile(values, 99)), 100 * expired / max(1, len(arrivals)), len(waits))
