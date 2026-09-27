from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from cnom.data.manifest import load_manifest, load_stream_manifest
from cnom.metrics.events import Event, match_events, temporal_iou
from cnom.utils import read_table, write_table


def build_dataset_report(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    section = config.get("dataset_report", {})
    dataset_cfg = config["dataset"]
    manifest = load_manifest(dataset_cfg["manifest"], root=dataset_cfg.get("root"), strict_paths=False)
    streams = load_stream_manifest(dataset_cfg, strict_paths=False)
    manifest["duration_s"] = manifest["end_s"].astype(float) - manifest["start_s"].astype(float)
    split_rows = []
    for split, group in manifest.groupby("split", dropna=False):
        stream_group = streams.loc[streams["split"].astype(str).eq(str(split))]
        worker_series = group.get("worker_proxy", group.get("worker_id", pd.Series(index=group.index, dtype=str)))
        split_rows.append({
            "split": split, "clips": len(group), "hours": float((stream_group["end_s"] - stream_group["start_s"]).sum() / 3600),
            "annotated_clip_hours": float(group["duration_s"].sum() / 3600), "sources": stream_group["source_id"].nunique(),
            "cameras": stream_group.get("camera_id", pd.Series(dtype=str)).replace("", pd.NA).nunique(),
            "scenes": stream_group.get("scene_id", pd.Series(dtype=str)).replace("", pd.NA).nunique(),
            "workers": worker_series.replace("", pd.NA).nunique(), "unsafe_clips": int((group["label"].astype(int) != 0).sum()),
            "safe_clips": int((group["label"].astype(int) == 0).sum()),
        })
    by_split = pd.DataFrame(split_rows)
    aggregations: dict[str, tuple[str, Any]] = {"clips": ("sample_id", "size"), "hours": ("duration_s", lambda value: value.sum() / 3600), "sources": ("source_id", "nunique")}
    if "occluded" in manifest.columns:
        aggregations["occluded_percent"] = ("occluded", lambda value: 100 * pd.Series(value).fillna(False).astype(bool).mean())
    if "ambiguous" in manifest.columns:
        aggregations["ambiguous_percent"] = ("ambiguous", lambda value: 100 * pd.Series(value).fillna(False).astype(bool).mean())
    by_class = manifest.groupby(["split", "label"], dropna=False).agg(**aggregations).reset_index()
    leakage = _group_leakage(manifest, tuple(config["dataset"].get("grouped_fields", ("source_id", "camera_id", "scene_id", "cluster_id"))))
    outputs = {"dataset_by_split": by_split, "dataset_by_class": by_class, "split_leakage": leakage}
    annotations = section.get("annotations")
    if annotations:
        outputs.update(annotation_agreement(read_table(annotations), section))
    if section.get("near_duplicate_pairs"):
        pairs = read_table(section["near_duplicate_pairs"])
        outputs["near_duplicate_pairs"] = pairs
        outputs["near_duplicate_summary"] = pd.DataFrame([{"pairs": len(pairs), "cross_split_pairs": int(pairs.get("cross_split", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())}])
    output_dir = Path(section.get("output_dir", Path(config.get("output_dir", "runs/default")) / "dataset_report"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        write_table(frame, output_dir / f"{name}.csv")
    return outputs


def annotation_agreement(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    required = {"annotator_id", "sample_id", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annotation table lacks columns: {sorted(missing)}")
    label_matrix = frame.pivot_table(index="sample_id", columns="annotator_id", values="label", aggfunc="first")
    rows = []
    for left, right in combinations(label_matrix.columns, 2):
        paired = label_matrix[[left, right]].dropna()
        rows.append({"annotator_a": left, "annotator_b": right, "shared_samples": len(paired), "cohen_kappa": cohen_kappa_score(paired[left], paired[right]) if len(paired) else np.nan})
    outputs = {"annotation_label_agreement": pd.DataFrame(rows)}
    event_columns = {"camera_id", "start_s", "end_s"}
    if event_columns.issubset(frame.columns):
        event_rows = []
        for left, right in combinations(sorted(frame["annotator_id"].astype(str).unique()), 2):
            left_events = _annotation_events(frame.loc[frame["annotator_id"].astype(str).eq(left)])
            right_events = _annotation_events(frame.loc[frame["annotator_id"].astype(str).eq(right)])
            matched = match_events(left_events, right_events, float(config.get("agreement_tiou", 0.3)), float(config.get("agreement_onset_tolerance_s", 2.0)), True)
            precision = matched.tp / (matched.tp + matched.fp) if matched.tp + matched.fp else np.nan
            recall = matched.tp / (matched.tp + matched.fn) if matched.tp + matched.fn else np.nan
            onset = [abs(left_events[item.truth_index].start_s - right_events[item.prediction_index].start_s) for item in matched.matches]
            offset = [abs(left_events[item.truth_index].end_s - right_events[item.prediction_index].end_s) for item in matched.matches]
            tiou = [temporal_iou(left_events[item.truth_index], right_events[item.prediction_index]) for item in matched.matches]
            event_rows.append({
                "annotator_a": left, "annotator_b": right, "events_a": len(left_events), "events_b": len(right_events),
                "matches": matched.tp, "precision": precision, "recall": recall,
                "median_onset_difference_s": float(np.median(onset)) if onset else np.nan,
                "median_offset_difference_s": float(np.median(offset)) if offset else np.nan,
                "median_tiou": float(np.median(tiou)) if tiou else np.nan,
            })
        outputs["annotation_event_agreement"] = pd.DataFrame(event_rows)
    if "adjudicated" in frame.columns:
        outputs["annotation_adjudication"] = frame.groupby("annotator_id", as_index=False).agg(samples=("sample_id", "size"), adjudicated=("adjudicated", lambda value: int(pd.Series(value).fillna(False).astype(bool).sum())))
        outputs["annotation_adjudication"]["adjudication_rate"] = outputs["annotation_adjudication"]["adjudicated"] / outputs["annotation_adjudication"]["samples"].clip(lower=1)
    return outputs


def _annotation_events(frame: pd.DataFrame) -> list[Event]:
    return [Event(str(row.sample_id), str(row.camera_id), float(row.start_s), float(row.end_s), int(row.label)) for row in frame.itertuples()]


def _group_leakage(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for field in fields:
        if field not in frame.columns:
            continue
        grouped = frame.dropna(subset=[field]).groupby(field)["split"].nunique()
        offending = grouped[grouped > 1]
        rows.append({"field": field, "groups": int(grouped.size), "cross_split_groups": int(offending.size), "clean": bool(offending.empty)})
    return pd.DataFrame(rows)
