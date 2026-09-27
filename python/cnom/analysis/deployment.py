from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cnom.metrics.statistics import wilson_interval
from cnom.utils import read_table, write_table


def analyze_deployment(config: dict[str, Any]) -> pd.DataFrame:
    section = config["deployment_analysis"]
    exposure = read_table(section["exposure"])
    alerts = read_table(section["alerts"])
    audit_events = read_table(section["audit_events"])
    audit_alerts = read_table(section.get("audit_alerts", section["alerts"]))
    incidents = read_table(section["incidents"]) if section.get("incidents") else pd.DataFrame()
    scheduled = float(exposure[section.get("scheduled_hours_column", "scheduled_camera_hours")].sum())
    valid = float(exposure[section.get("valid_hours_column", "valid_camera_hours")].sum())
    verified = alerts[section.get("verified_column", "verified_unsafe")].fillna(False).astype(bool)
    alert_tp, alert_fp = int(verified.sum()), int((~verified).sum())
    detected = audit_events[section.get("detected_column", "detected")].fillna(False).astype(bool)
    audit_tp, audit_fn = int(detected.sum()), int((~detected).sum())
    audit_verified = audit_alerts[section.get("verified_column", "verified_unsafe")].fillna(False).astype(bool)
    audit_fp = int((~audit_verified).sum())
    rows = [
        _metric("scheduled_camera_hours", scheduled, "camera_hours"), _metric("valid_camera_hours", valid, "camera_hours"),
        _metric("availability", valid / scheduled if scheduled else 0.0, "proportion", *wilson_interval(round(valid), round(scheduled)) if scheduled and scheduled.is_integer() and valid.is_integer() else (None, None)),
        _metric("production_alerts", len(alerts), "count"), _metric("verified_unsafe_alerts", alert_tp, "count"), _metric("false_alerts", alert_fp, "count"),
        _metric("production_alert_precision", alert_tp / max(1, alert_tp + alert_fp), "proportion", *wilson_interval(alert_tp, alert_tp + alert_fp)),
        _metric("false_alerts_per_camera_hour", alert_fp / max(valid, 1e-9), "rate"),
        _metric("audit_reference_events", audit_tp + audit_fn, "count"), _metric("audit_true_positives", audit_tp, "count"), _metric("audit_false_negatives", audit_fn, "count"),
        _metric("audit_recall", audit_tp / max(1, audit_tp + audit_fn), "proportion", *wilson_interval(audit_tp, audit_tp + audit_fn)),
        _metric("audit_false_positives", audit_fp, "count"),
        _metric("audit_precision", audit_tp / max(1, audit_tp + audit_fp), "proportion", *wilson_interval(audit_tp, audit_tp + audit_fp)),
    ]
    if not incidents.empty:
        queued = float(incidents.get("clips_queued", pd.Series(dtype=float)).fillna(0).sum())
        recovered = float(incidents.get("clips_recovered", pd.Series(dtype=float)).fillna(0).sum())
        expired = float(incidents.get("clips_expired", pd.Series(dtype=float)).fillna(0).sum())
        lost = float(incidents.get("clips_lost", pd.Series(dtype=float)).fillna(0).sum())
        rows.extend([_metric("infrastructure_incidents", len(incidents), "count"), _metric("queued_clips_during_incidents", queued, "count"), _metric("recovered_clips", recovered, "count"), _metric("expired_clips", expired, "count"), _metric("lost_clips", lost, "count"), _metric("queue_recovery_rate", recovered / max(1.0, queued), "proportion")])
    result = pd.DataFrame(rows)
    output_dir = Path(config.get("output_dir", "runs/deployment_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(result, output_dir / "deployment_evidence.csv")
    summary = {row["metric"]: row["value"] for row in rows}
    (output_dir / "deployment_evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def _metric(name: str, value: float, unit: str, ci_low: float | None = None, ci_high: float | None = None) -> dict[str, Any]:
    return {"metric": name, "value": value, "unit": unit, "ci_low": ci_low, "ci_high": ci_high}
