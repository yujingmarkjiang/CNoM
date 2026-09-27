from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from cnom.utils import read_table, write_table


def analyze_drift(config: dict[str, Any]) -> pd.DataFrame:
    section = config["drift"]
    reference, current = read_table(section["reference"]), read_table(section["current"])
    features = list(section.get("features") or _numeric_intersection(reference, current))
    if not features:
        raise ValueError("No common numeric drift features were selected")
    rows: list[dict[str, Any]] = []
    bins = int(section.get("psi_bins", 10))
    for feature in features:
        ref, cur = reference[feature].dropna().to_numpy(float), current[feature].dropna().to_numpy(float)
        rows.append({
            "feature": feature, "reference_n": len(ref), "current_n": len(cur),
            "reference_mean": np.mean(ref) if len(ref) else np.nan, "current_mean": np.mean(cur) if len(cur) else np.nan,
            "standardized_mean_shift": _standardized_shift(ref, cur), "psi": _psi(ref, cur, bins),
        })
    frame = pd.DataFrame(rows)
    if bool(section.get("compute_mmd", True)):
        sample = int(section.get("mmd_sample", 2000))
        frame.attrs["mmd_rbf"] = _mmd(reference[features].dropna().to_numpy(float), current[features].dropna().to_numpy(float), sample, int(config.get("seed", 2027)))
    output = Path(section.get("output", Path(config.get("output_dir", "runs/default")) / "drift.csv"))
    write_table(frame, output)
    output.with_suffix(".summary.json").write_text(pd.Series({"mmd_rbf": frame.attrs.get("mmd_rbf")}).to_json(indent=2), encoding="utf-8")
    return frame


def _numeric_intersection(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return [column for column in left.select_dtypes(include="number").columns if column in right.columns]


def _standardized_shift(reference: np.ndarray, current: np.ndarray) -> float:
    if not len(reference) or not len(current):
        return float("nan")
    scale = np.std(reference, ddof=1)
    return float((np.mean(current) - np.mean(reference)) / scale) if scale > 0 else 0.0


def _psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    if not len(reference) or not len(current):
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _mmd(reference: np.ndarray, current: np.ndarray, sample: int, seed: int) -> float:
    if not len(reference) or not len(current):
        return float("nan")
    rng = np.random.default_rng(seed)
    reference = reference[rng.choice(len(reference), min(sample, len(reference)), replace=False)]
    current = current[rng.choice(len(current), min(sample, len(current)), replace=False)]
    stacked = np.vstack([reference, current])
    scale = np.nanstd(stacked, axis=0)
    scale[scale == 0] = 1
    reference, current = (reference - np.nanmean(stacked, axis=0)) / scale, (current - np.nanmean(stacked, axis=0)) / scale
    distances = cdist(stacked[: min(500, len(stacked))], stacked[: min(500, len(stacked))], "sqeuclidean")
    positive = distances[distances > 0]
    gamma = 1 / max(float(np.median(positive)), 1e-6)
    kxx = np.exp(-gamma * cdist(reference, reference, "sqeuclidean"))
    kyy = np.exp(-gamma * cdist(current, current, "sqeuclidean"))
    kxy = np.exp(-gamma * cdist(reference, current, "sqeuclidean"))
    return float(max(0.0, kxx.mean() + kyy.mean() - 2 * kxy.mean()))
