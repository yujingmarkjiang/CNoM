from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cnom.utils import read_table, write_table


def analyze_failures(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    events = read_table(config["events"])
    alerts = read_table(config["alerts"])
    incidents = read_table(config["incidents"])
    reviews = read_table(config["reviews"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    detected = events["detected"].astype(bool) if "detected" in events.columns else pd.Series(False, index=events.index)
    misses = events.loc[~detected].copy()
    if "miss_stage" not in misses.columns:
        misses["miss_stage"] = misses.apply(_derive_miss_stage, axis=1)
    miss_summary = misses.groupby(["miss_stage", "label"], dropna=False).size().rename("count").reset_index()
    miss_totals = misses.groupby("miss_stage", dropna=False).size().rename("count").reset_index()
    miss_totals["share"] = miss_totals["count"] / max(1, miss_totals["count"].sum())
    verified = alerts["verified_unsafe"].astype(bool) if "verified_unsafe" in alerts.columns else pd.Series(True, index=alerts.index)
    false_alerts = alerts.loc[~verified].copy()
    reason_column = "disposition_reason" if "disposition_reason" in false_alerts.columns else "root_cause"
    false_alert_summary = false_alerts.groupby(reason_column, dropna=False).size().rename("count").reset_index() if reason_column in false_alerts.columns else pd.DataFrame()
    disagreement = reviews.loc[reviews.get("model_label", -1) != reviews.get("reviewer_label", -1)].copy() if {"model_label", "reviewer_label"}.issubset(reviews.columns) else pd.DataFrame()
    incident_summary = incidents.groupby("category", dropna=False).agg(incidents=("category", "size"), outage_minutes=("duration_minutes", "sum"), clips_queued=("clips_queued", "sum"), clips_recovered=("clips_recovered", "sum"), clips_expired=("clips_expired", "sum")).reset_index()
    outputs = {
        "miss_root_causes": miss_totals, "miss_by_class": miss_summary, "false_alert_reasons": false_alert_summary,
        "reviewer_disagreements": disagreement, "incident_summary": incident_summary,
    }
    for name, frame in outputs.items():
        write_table(frame, output_dir / f"{name}.csv")
    return outputs


def _derive_miss_stage(row: pd.Series) -> str:
    if not bool(row.get("sentinel_above_threshold", False)):
        return "no_sentinel_score"
    if not bool(row.get("persistence_passed", False)):
        return "insufficient_persistence"
    if not bool(row.get("clip_constructed", True)):
        return "clip_construction_or_encoding"
    if not bool(row.get("cloud_received", True)):
        return "network_queue_or_expiration"
    if int(row.get("teacher_label", 0)) == 0:
        return "teacher_unsafe_to_safe"
    if bool(row.get("suppressed", False)):
        return "suppression_policy"
    if not bool(row.get("dashboard_delivered", True)):
        return "dashboard_delivery"
    return "annotation_or_ontology_disagreement"
