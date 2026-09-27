from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from cnom.utils import read_table, write_table


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    bandwidth_mbps: float
    rtt_ms: float
    jitter_ms: float
    packet_loss_percent: float
    outage_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or self.bandwidth_mbps <= 0:
            raise ValueError("Network profiles require a name and positive bandwidth")
        if min(self.rtt_ms, self.jitter_ms, self.outage_s) < 0 or not 0 <= self.packet_loss_percent <= 100:
            raise ValueError("Network delays must be non-negative and packet loss must be within [0, 100]")


class NetemContext:
    def __init__(self, interface: str, profile: NetworkProfile, enabled: bool) -> None:
        self.interface, self.profile, self.enabled = interface, profile, enabled

    def __enter__(self):
        if self.enabled:
            rate = max(self.profile.bandwidth_mbps, 0.01)
            command = f"tc qdisc replace dev {shlex.quote(self.interface)} root netem delay {self.profile.rtt_ms}ms {self.profile.jitter_ms}ms loss {self.profile.packet_loss_percent}% rate {rate}mbit"
            subprocess.run(command, shell=True, check=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.enabled:
            subprocess.run(["tc", "qdisc", "del", "dev", self.interface, "root"], check=False)


async def run_network_stress(config: dict[str, Any]) -> pd.DataFrame:
    stress = config["stress"]
    manifest = read_table(stress["manifest"])
    profiles = [NetworkProfile(**profile) for profile in stress["profiles"]]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        with NetemContext(str(stress.get("tc_interface", "eth0")), profile, bool(stress.get("apply_tc", False))):
            rows.append(await _replay_profile(manifest, stress, profile, int(config.get("seed", 2027))))
    result = pd.DataFrame(rows)
    write_table(result, output_dir / "network_stress.csv")
    return result


async def _replay_profile(manifest: pd.DataFrame, config: dict[str, Any], profile: NetworkProfile, seed: int) -> dict[str, Any]:
    key = os.environ.get(str(config.get("api_key_env", "CNOM_API_KEY")), "")
    concurrency = int(config.get("concurrency", 24))
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    expired, recovered = 0, 0
    repetitions = int(config.get("repetitions", 5))
    maximum_age = float(config.get("max_event_age_s", 1800))
    if repetitions <= 0 or maximum_age <= 0:
        raise ValueError("repetitions and max_event_age_s must be positive")
    apply_tc = bool(config.get("apply_tc", False))
    verify: Any = config.get("ca_bundle") or bool(config.get("verify_tls", True))
    rng = np.random.default_rng(seed + sum(map(ord, profile.name)))
    lock = asyncio.Lock()
    timeout = httpx.Timeout(float(config.get("request_timeout_s", 180)), connect=float(config.get("connect_timeout_s", 10)))
    async with httpx.AsyncClient(timeout=timeout, verify=verify, http2=bool(config.get("http2", True))) as client:
        async def send(row, repetition: int) -> None:
            nonlocal expired, recovered
            async with semaphore:
                path = Path(row.video_path)
                payload = path.read_bytes()
                metadata = _metadata(row)
                simulated = _simulated_delay(len(payload), profile, rng) if not apply_tc else 0.0
                started = time.perf_counter()
                if profile.outage_s and not apply_tc:
                    simulated += profile.outage_s
                    recovered += 1
                if simulated > maximum_age:
                    expired += 1
                    return
                if not apply_tc and profile.packet_loss_percent < 100 and rng.random() < profile.packet_loss_percent / 100:
                    simulated += float(config.get("retry_backoff_s", 1.0))
                    recovered += 1
                if bool(config.get("send_requests", True)):
                    digest = hashlib.sha256(payload).hexdigest()
                    response = await client.post(
                        str(config["target_url"]),
                        headers={"X-API-Key": key, "X-Content-SHA256": digest, "X-Idempotency-Key": f"stress:{profile.name}:{repetition}:{row.sample_id}"},
                        files={"file": (path.name, payload, "video/mp4")}, data={"metadata": json.dumps(metadata)},
                    )
                    response.raise_for_status()
                latency = time.perf_counter() - started + simulated
                async with lock:
                    latencies.append(latency)

        tasks = [asyncio.create_task(send(row, repetition)) for repetition in range(repetitions) for _, row in manifest.iterrows()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = sum(isinstance(result, Exception) for result in results)
    total = len(manifest) * repetitions
    expired += failures
    values = np.asarray(latencies or [maximum_age], dtype=float)
    return {
        "condition": profile.name, "link_mbps": profile.bandwidth_mbps, "rtt_ms": profile.rtt_ms, "jitter_ms": profile.jitter_ms,
        "packet_loss_percent": profile.packet_loss_percent, "concurrent_streams": int(config.get("concurrent_streams", config.get("concurrency", 24))),
        "cloud_workers": int(config.get("cloud_workers", 1)), "events": total, "repetitions": repetitions,
        "median_latency_s": float(np.median(values)), "p95_latency_s": float(np.percentile(values, 95)), "p99_latency_s": float(np.percentile(values, 99)),
        "expired_percent": 100 * expired / max(1, total), "recovered_percent": 100 * recovered / max(1, total),
        "notes": "real netem replay" if apply_tc else "deterministic impairment simulation with optional API submission",
    }


def _simulated_delay(size_bytes: int, profile: NetworkProfile, rng: np.random.Generator) -> float:
    bandwidth = max(profile.bandwidth_mbps, 0.001) * 1e6
    transfer = size_bytes * 8 / bandwidth
    jitter = rng.normal(0, profile.jitter_ms / 1000) if profile.jitter_ms else 0.0
    return max(0.0, profile.rtt_ms / 1000 + jitter + transfer)


def _metadata(row: pd.Series) -> dict[str, Any]:
    start, end = float(row.get("start_s", 0)), float(row.get("end_s", 1))
    return {
        "candidate_id": str(row.get("candidate_id", row.get("sample_id"))), "camera_id": str(row.get("camera_id", "stress_camera")),
        "source_id": str(row.get("source_id", row.get("sample_id"))), "zone_id": row.get("zone_id"),
        "start_s": start, "end_s": max(end, start + 0.001), "trigger_s": float(row.get("trigger_s", start)),
        "edge_max_score": float(row.get("edge_max_score", 1.0)), "edge_mean_score": float(row.get("edge_mean_score", 1.0)),
    }
