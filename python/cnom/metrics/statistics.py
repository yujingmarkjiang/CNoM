from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if trials == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 2027,
) -> tuple[float, float]:
    values = np.asarray(values)
    if not 0 < confidence < 1 or repetitions <= 0:
        raise ValueError("confidence must be between 0 and 1 and repetitions must be positive")
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates[index] = statistic(sample)
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def cluster_bootstrap_ci(
    frame,
    cluster_column: str,
    statistic: Callable,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 2027,
) -> tuple[float, float]:
    if cluster_column not in frame.columns:
        raise ValueError(f"Missing cluster column: {cluster_column}")
    if not 0 < confidence < 1 or repetitions <= 0:
        raise ValueError("confidence must be between 0 and 1 and repetitions must be positive")
    clusters = np.asarray(frame[cluster_column].dropna().unique())
    if len(clusters) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    grouped = {cluster: frame.loc[frame[cluster_column] == cluster] for cluster in clusters}
    for index in range(repetitions):
        sampled = clusters[rng.integers(0, len(clusters), len(clusters))]
        parts = [grouped[cluster].assign(_bootstrap_cluster=f"{position}:{cluster}") for position, cluster in enumerate(sampled)]
        estimates[index] = statistic(__import__("pandas").concat(parts, ignore_index=True))
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))
