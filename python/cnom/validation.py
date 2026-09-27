from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cnom.data.manifest import audit_split_leakage, load_manifest
from cnom.utils import read_table


def validate_manifest_file(path: str | Path, root: str | Path | None = None, grouped_fields: tuple[str, ...] = ("source_id", "camera_id", "scene_id", "cluster_id"), strict_paths: bool = True) -> dict[str, Any]:
    frame = load_manifest(path, root=root, strict_paths=strict_paths)
    audit = audit_split_leakage(frame, grouped_fields)
    report = {
        "samples": len(frame), "sources": int(frame["source_id"].nunique()), "splits": frame["split"].value_counts().to_dict(),
        "classes": frame["label"].value_counts().sort_index().to_dict(), "duration_hours": float((frame["end_s"] - frame["start_s"]).sum() / 3600),
        "leakage": audit.overlap_by_field, "invalid_intervals": audit.invalid_intervals, "missing_files": audit.missing_files,
        "duplicated_sample_ids": audit.duplicated_sample_ids, "passed": audit.passed,
    }
    if not audit.passed:
        raise ValueError(json.dumps(report, indent=2))
    return report


def validate_result_directory(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for file in sorted(root.rglob("*")):
        if file.suffix.lower() not in {".csv", ".parquet", ".json"}:
            continue
        try:
            if file.suffix.lower() == ".json":
                payload = json.loads(file.read_text(encoding="utf-8"))
                _scan_json(payload, str(file.relative_to(root)), errors)
                checks.append({"file": str(file.relative_to(root)), "type": "json", "status": "passed"})
            else:
                frame = read_table(file)
                _validate_table(frame, str(file.relative_to(root)), errors)
                checks.append({"file": str(file.relative_to(root)), "type": "table", "rows": len(frame), "status": "passed"})
        except Exception as exc:
            errors.append(f"{file.relative_to(root)}: {exc}")
    report = {"root": str(root), "files_checked": len(checks), "checks": checks, "errors": errors, "passed": not errors}
    (root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError("Result validation failed:\n" + "\n".join(errors))
    return report


def _validate_table(frame: pd.DataFrame, name: str, errors: list[str]) -> None:
    if frame.empty:
        errors.append(f"{name}: table is empty")
        return
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        errors.append(f"{name}: contains infinite numeric values")
    probability_columns = [column for column in frame.columns if column.startswith("p_") or any(token in column.lower() for token in ("recall", "precision", "f1", "accuracy", "availability"))]
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty and ((values < 0) | (values > 1)).any() and not column.lower().endswith("percent"):
            errors.append(f"{name}: {column} is outside [0, 1]")
    if {"tp", "fn", "recall"}.issubset(frame.columns):
        expected = frame["tp"] / (frame["tp"] + frame["fn"]).replace(0, np.nan)
        _compare(expected, frame["recall"], f"{name}: recall", errors)
    if {"tp", "fp", "precision"}.issubset(frame.columns):
        expected = frame["tp"] / (frame["tp"] + frame["fp"]).replace(0, np.nan)
        _compare(expected, frame["precision"], f"{name}: precision", errors)
    if {"event_tp", "event_fn", "reported_recall"}.issubset(frame.columns):
        expected = frame["event_tp"] / (frame["event_tp"] + frame["event_fn"]).replace(0, np.nan)
        _compare(expected, frame["reported_recall"], f"{name}: reported_recall", errors)
    if {"event_tp", "event_fp", "reported_precision"}.issubset(frame.columns):
        expected = frame["event_tp"] / (frame["event_tp"] + frame["event_fp"]).replace(0, np.nan)
        _compare(expected, frame["reported_precision"], f"{name}: reported_precision", errors)
    if {"baseline_value", "deployed_value", "relative_change_percent"}.issubset(frame.columns):
        expected = 100 * (frame["deployed_value"] - frame["baseline_value"]) / frame["baseline_value"].replace(0, np.nan)
        _compare(expected, frame["relative_change_percent"], f"{name}: relative_change_percent", errors, tolerance=1e-4)
    if {"median_latency_s", "p95_latency_s"}.issubset(frame.columns):
        invalid = frame["p95_latency_s"].notna() & frame["median_latency_s"].notna() & (frame["p95_latency_s"] < frame["median_latency_s"])
        if invalid.any():
            errors.append(f"{name}: p95 latency is below median latency")


def _scan_json(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _scan_json(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_json(item, f"{path}[{index}]", errors)
    elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        errors.append(f"{path}: non-finite value")


def _compare(expected: pd.Series, observed: pd.Series, label: str, errors: list[str], tolerance: float = 1e-6) -> None:
    mask = expected.notna() & pd.to_numeric(observed, errors="coerce").notna()
    if mask.any() and not np.allclose(expected[mask].astype(float), pd.to_numeric(observed[mask]).astype(float), atol=tolerance, rtol=tolerance):
        errors.append(f"{label} is inconsistent with count columns")
