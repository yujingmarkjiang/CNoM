from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import numpy as np


class RouteDecision(StrEnum):
    SUPPRESSED_SAFE = "suppressed_safe"
    UNSAFE_ALERT = "unsafe_alert"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class RouterConfig:
    safe_threshold: float = 0.985
    unsafe_threshold: float = 0.80
    non_suppressible_classes: tuple[int, ...] = (3, 6)


@dataclass(frozen=True)
class RoutedPrediction:
    decision: RouteDecision
    label: int
    confidence: float
    safe_probability: float
    unsafe_probability: float


class Router:
    def __init__(self, config: RouterConfig) -> None:
        if not 0 <= config.safe_threshold <= 1 or not 0 <= config.unsafe_threshold <= 1:
            raise ValueError("Routing thresholds must be within [0, 1]")
        if any(int(value) <= 0 for value in config.non_suppressible_classes):
            raise ValueError("non_suppressible_classes must contain positive unsafe class IDs")
        self.config = config

    def route(self, probabilities: np.ndarray) -> RoutedPrediction:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or len(probabilities) < 2 or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Probabilities must be a finite non-negative vector containing safe and unsafe classes")
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("Probability vector has no positive mass")
        probabilities = probabilities / total
        safe_probability = float(probabilities[0])
        label = int(np.argmax(probabilities))
        unsafe_probability = float(1 - safe_probability)
        confidence = float(probabilities[label])
        if unsafe_probability >= self.config.unsafe_threshold:
            decision = RouteDecision.UNSAFE_ALERT
        elif safe_probability >= self.config.safe_threshold and label not in self.config.non_suppressible_classes:
            decision = RouteDecision.SUPPRESSED_SAFE
        else:
            decision = RouteDecision.MANUAL_REVIEW
        return RoutedPrediction(decision, label, confidence, safe_probability, unsafe_probability)


def resolve_class_ids(values: Iterable[int | str], labels: Iterable[str]) -> tuple[int, ...]:
    label_names = tuple(str(name) for name in labels)
    label_map = {name: index for index, name in enumerate(label_names)}
    resolved: list[int] = []
    for value in values:
        if isinstance(value, int) or str(value).isdigit():
            class_id = int(value)
            if class_id < 0 or class_id >= len(label_names):
                raise ValueError(f"Class identifier is out of range: {value}")
            resolved.append(class_id)
        elif str(value) in label_map:
            resolved.append(label_map[str(value)])
        else:
            raise ValueError(f"Unknown class identifier: {value}")
    return tuple(sorted(set(resolved)))
