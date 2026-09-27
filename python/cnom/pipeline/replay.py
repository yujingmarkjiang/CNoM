from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from cnom.metrics.events import Event, evaluate_events
from cnom.pipeline.routing import RouteDecision, Router
from cnom.pipeline.temporal import TemporalPolicy, generate_candidates

TeacherScorer = Callable[[pd.DataFrame], np.ndarray]


def replay_cascade(
    frame_scores: pd.DataFrame,
    truth: list[Event],
    teacher_scorer: TeacherScorer,
    temporal_policy: TemporalPolicy,
    router: Router,
    camera_hours: float,
    manual_review_labels: dict[str, int] | None = None,
    evaluation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = generate_candidates(frame_scores, temporal_policy)
    if candidates.empty:
        metrics = evaluate_events(truth, [], camera_hours, num_classes=int((evaluation or {}).get("num_classes", 9)))
        metrics.update({"candidate_events": 0, "uploaded_duration_s": 0.0, "manual_review_items": 0, "suppressed_safe_items": 0})
        return metrics, candidates
    probabilities = teacher_scorer(candidates)
    if probabilities.shape[0] != len(candidates):
        raise ValueError("Teacher scorer returned an unexpected number of rows")
    rows: list[dict[str, Any]] = []
    predictions: list[Event] = []
    for candidate, probability in zip(candidates.itertuples(), probabilities, strict=True):
        routed = router.route(probability)
        label = routed.label
        if routed.decision == RouteDecision.MANUAL_REVIEW and manual_review_labels is not None:
            label = int(manual_review_labels.get(candidate.candidate_id, label))
        is_alert = routed.decision == RouteDecision.UNSAFE_ALERT or (routed.decision == RouteDecision.MANUAL_REVIEW and label != 0)
        row = {**candidate._asdict(), "route": routed.decision.value, "label": label, "confidence": routed.confidence, "safe_probability": routed.safe_probability, "unsafe_probability": routed.unsafe_probability}
        rows.append(row)
        if is_alert:
            predictions.append(Event(candidate.candidate_id, candidate.camera_id, candidate.start_s, candidate.end_s, label, routed.confidence, candidate.source_id))
    evaluation = evaluation or {}
    metrics = evaluate_events(
        truth, predictions, camera_hours, tiou_threshold=float(evaluation.get("tiou_threshold", 0.3)),
        onset_tolerance_s=float(evaluation.get("onset_tolerance_s", 2.0)), class_aware=bool(evaluation.get("class_aware_matching", True)),
        num_classes=int(evaluation.get("num_classes", probabilities.shape[1])),
    )
    metrics["candidate_events"] = len(candidates)
    metrics["uploaded_duration_s"] = float((candidates["end_s"] - candidates["start_s"]).sum())
    metrics["manual_review_items"] = sum(row["route"] == RouteDecision.MANUAL_REVIEW.value for row in rows)
    metrics["suppressed_safe_items"] = sum(row["route"] == RouteDecision.SUPPRESSED_SAFE.value for row in rows)
    return metrics, pd.DataFrame(rows)
