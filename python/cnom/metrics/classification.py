from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: int | Sequence[int] | np.ndarray) -> list[dict[str, int]]:
    label_values = np.arange(labels) if isinstance(labels, int) else np.asarray(labels, dtype=int)
    observed = set(np.asarray(y_true, dtype=int)).union(np.asarray(y_pred, dtype=int))
    if not observed.issubset(set(label_values.tolist())):
        raise ValueError("Configured labels do not cover every observed class")
    matrix = confusion_matrix(y_true, y_pred, labels=label_values)
    rows: list[dict[str, int]] = []
    total = int(matrix.sum())
    for index, label in enumerate(label_values):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        rows.append({"label": int(label), "tp": tp, "fn": fn, "fp": fp, "tn": total - tp - fn - fp})
    return rows


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if len(y_true) == 0:
        raise ValueError("Classification metrics require at least one sample")
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Classification labels must be non-negative integers")
    if num_classes is not None and num_classes <= 0:
        raise ValueError("num_classes must be positive")
    labels = np.arange(num_classes, dtype=int) if num_classes is not None else np.unique(np.concatenate([y_true, y_pred]))
    if not set(np.concatenate([y_true, y_pred]).tolist()).issubset(set(labels.tolist())):
        raise ValueError("Observed labels fall outside num_classes")
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_labels": labels.tolist(),
        "per_class_precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "counts": confusion_counts(y_true, y_pred, labels),
    }
    if probabilities is not None:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(y_true):
            raise ValueError("probabilities must have shape [samples, classes]")
        if num_classes is not None and values.shape[1] != num_classes:
            raise ValueError("Probability columns must match num_classes")
        if values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("Probabilities must be finite and contain at least one class")
        result["mean_confidence"] = float(values.max(axis=1).mean())
    return result
