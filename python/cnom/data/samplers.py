from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


def sample_low_label(frame: pd.DataFrame, budget_per_class: int, seed: int, group_field: str = "source_id", safe_ratio: float = 1.0) -> pd.DataFrame:
    if budget_per_class < 0 or safe_ratio < 0:
        raise ValueError("Label budgets and safe_ratio must be non-negative")
    rng = np.random.default_rng(seed)
    train = frame.loc[frame["split"].astype(str) == "train"].copy()
    fixed = frame.loc[frame["split"].astype(str) != "train"].copy()
    chosen: list[pd.DataFrame] = []
    unsafe_classes = sorted(label for label in train["label"].unique() if int(label) != 0)
    for label in unsafe_classes:
        chosen.append(_group_sample(train.loc[train["label"] == label], budget_per_class, group_field, rng))
    safe_budget = int(round(budget_per_class * max(1, len(unsafe_classes)) * safe_ratio))
    chosen.append(_group_sample(train.loc[train["label"] == 0], safe_budget, group_field, rng))
    sampled = pd.concat(chosen + [fixed], ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def _group_sample(frame: pd.DataFrame, count: int, group_field: str, rng: np.random.Generator) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= count:
        return frame.copy()
    if group_field not in frame.columns:
        return frame.iloc[rng.choice(len(frame), count, replace=False)].copy()
    groups = frame[group_field].dropna().unique().tolist()
    rng.shuffle(groups)
    selected: list[int] = []
    for group in groups:
        indices = frame.index[frame[group_field] == group].tolist()
        rng.shuffle(indices)
        selected.extend(indices[: max(0, count - len(selected))])
        if len(selected) >= count:
            break
    if len(selected) < count:
        remaining = frame.index.difference(selected).to_numpy()
        selected.extend(rng.choice(remaining, count - len(selected), replace=False).tolist())
    return frame.loc[selected[:count]].copy()


class DistributedWeightedSampler(Sampler[int]):
    def __init__(self, weights: torch.Tensor, num_replicas: int, rank: int, replacement: bool = True, seed: int = 0) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler rank or world size")
        self.weights = weights.double().reshape(-1)
        if len(self.weights) == 0 or not torch.isfinite(self.weights).all() or (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("Sampling weights must be finite, non-negative, non-empty, and have positive mass")
        self.num_replicas, self.rank, self.replacement, self.seed = num_replicas, rank, replacement, seed
        self.num_samples = math.ceil(len(weights) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=generator).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without duplicating samples to equalize rank lengths."""

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler rank or world size")
        self.length, self.num_replicas, self.rank = len(dataset), num_replicas, rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.num_replicas))

    def __len__(self) -> int:
        return max(0, (self.length - self.rank + self.num_replicas - 1) // self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        return None
