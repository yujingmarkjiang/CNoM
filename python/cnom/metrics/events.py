from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cnom.metrics.statistics import wilson_interval


@dataclass(frozen=True)
class Event:
    event_id: str
    camera_id: str
    start_s: float
    end_s: float
    label: int
    score: float = 1.0
    cluster_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id or not self.camera_id:
            raise ValueError("Event identifiers and camera_id must be non-empty")
        if not np.isfinite([self.start_s, self.end_s, self.score]).all() or self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("Event intervals must be finite and satisfy 0 <= start_s < end_s")
        if self.label < 0:
            raise ValueError("Event labels must be non-negative")

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class EventMatch:
    truth_index: int
    prediction_index: int
    weight: float


@dataclass
class EventMatchResult:
    matches: list[EventMatch]
    unmatched_truth: list[int]
    unmatched_predictions: list[int]

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fn(self) -> int:
        return len(self.unmatched_truth)

    @property
    def fp(self) -> int:
        return len(self.unmatched_predictions)


def temporal_iou(left: Event, right: Event) -> float:
    intersection = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    union = max(left.end_s, right.end_s) - min(left.start_s, right.start_s)
    return intersection / union if union > 0 else 0.0


def match_events(
    truth: Iterable[Event],
    predictions: Iterable[Event],
    tiou_threshold: float = 0.3,
    onset_tolerance_s: float = 2.0,
    class_aware: bool = True,
) -> EventMatchResult:
    if not 0 <= tiou_threshold <= 1 or onset_tolerance_s < 0:
        raise ValueError("tiou_threshold must be in [0, 1] and onset_tolerance_s must be non-negative")
    truth, predictions = list(truth), list(predictions)
    if not truth:
        return EventMatchResult([], [], list(range(len(predictions))))
    if not predictions:
        return EventMatchResult([], list(range(len(truth))), [])
    weights = np.zeros((len(truth), len(predictions)), dtype=float)
    for i, actual in enumerate(truth):
        for j, predicted in enumerate(predictions):
            if actual.camera_id != predicted.camera_id:
                continue
            if class_aware and actual.label != predicted.label:
                continue
            tiou = temporal_iou(actual, predicted)
            overlaps = max(actual.start_s, predicted.start_s) < min(actual.end_s, predicted.end_s)
            onset_ok = overlaps and abs(actual.start_s - predicted.start_s) <= onset_tolerance_s
            if tiou >= tiou_threshold or onset_ok:
                weights[i, j] = max(tiou, 1.0 / (1.0 + abs(actual.start_s - predicted.start_s)))
    row_ind, col_ind = linear_sum_assignment(-weights)
    matches = [EventMatch(int(i), int(j), float(weights[i, j])) for i, j in zip(row_ind, col_ind, strict=True) if weights[i, j] > 0]
    matched_truth = {match.truth_index for match in matches}
    matched_predictions = {match.prediction_index for match in matches}
    return EventMatchResult(matches, [i for i in range(len(truth)) if i not in matched_truth], [j for j in range(len(predictions)) if j not in matched_predictions])


def recall_decomposition(p_escalation: float, p_alert: float, p_human: float, reviewer_recall: float) -> float:
    return p_escalation * (p_alert + p_human * reviewer_recall)


def evaluate_events(
    truth: Iterable[Event],
    predictions: Iterable[Event],
    camera_hours: float,
    tiou_threshold: float = 0.3,
    onset_tolerance_s: float = 2.0,
    class_aware: bool = True,
    num_classes: int = 9,
) -> dict[str, object]:
    if camera_hours < 0 or num_classes <= 1:
        raise ValueError("camera_hours must be non-negative and num_classes must exceed one")
    truth, predictions = list(truth), list(predictions)
    if any(event.label >= num_classes for event in [*truth, *predictions]):
        raise ValueError("Event labels fall outside num_classes")
    result = match_events(truth, predictions, tiou_threshold, onset_tolerance_s, class_aware)
    recall = result.tp / (result.tp + result.fn) if result.tp + result.fn else float("nan")
    precision = result.tp / (result.tp + result.fp) if result.tp + result.fp else float("nan")
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    recall_ci = wilson_interval(result.tp, result.tp + result.fn)
    precision_ci = wilson_interval(result.tp, result.tp + result.fp)
    per_class: list[dict[str, object]] = []
    for label in range(1, num_classes):
        class_truth = [event for event in truth if event.label == label]
        class_predictions = [event for event in predictions if event.label == label]
        class_result = match_events(class_truth, class_predictions, tiou_threshold, onset_tolerance_s, True)
        class_recall = class_result.tp / len(class_truth) if class_truth else float("nan")
        class_precision = class_result.tp / (class_result.tp + class_result.fp) if class_result.tp + class_result.fp else float("nan")
        class_f1 = 2 * class_recall * class_precision / (class_recall + class_precision) if class_recall + class_precision else 0.0
        per_class.append({"label": label, "events": len(class_truth), "tp": class_result.tp, "fn": class_result.fn, "fp": class_result.fp, "recall": class_recall, "precision": class_precision, "f1": class_f1})
    macro_f1 = float(np.nanmean([row["f1"] for row in per_class]))
    return {
        "tp": result.tp, "fn": result.fn, "fp": result.fp, "recall": recall, "recall_ci": recall_ci,
        "precision": precision, "precision_ci": precision_ci, "f1": f1, "macro_f1": macro_f1,
        "false_alerts_per_camera_hour": result.fp / camera_hours if camera_hours > 0 else float("nan"),
        "per_class": per_class, "matches": result.matches,
    }


def events_from_frame(frame: pd.DataFrame, score_column: str = "score") -> list[Event]:
    required = {"camera_id", "start_s", "end_s", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Event table lacks columns: {sorted(missing)}")
    events: list[Event] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        event_id = getattr(row, "event_id", None) or f"event-{index:08d}"
        events.append(Event(
            event_id=str(event_id), camera_id=str(getattr(row, "camera_id")),
            start_s=float(getattr(row, "start_s")), end_s=float(getattr(row, "end_s")),
            label=int(getattr(row, "label")), score=float(getattr(row, score_column, 1.0)),
            cluster_id=str(getattr(row, "cluster_id", "")),
        ))
    return events
