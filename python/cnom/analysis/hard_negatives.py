from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cnom.data.manifest import load_manifest, save_manifest
from cnom.inference import build_clip_loader, predict_dataset
from cnom.models.registry import build_model
from cnom.utils import resolve_device


def mine_hard_negatives(config: dict[str, Any]) -> pd.DataFrame:
    section = config["hard_negative_mining"]
    split = str(section.get("split", config["dataset"].get("train_split", "train")))
    model_cfg = dict(section.get("model", config.get("sentinel", config.get("model", {}))))
    model = build_model(model_cfg, checkpoint=section.get("checkpoint", model_cfg.get("checkpoint")), strict=False)
    local = {**config, "model": model_cfg}
    predictions, _ = predict_dataset(model, build_clip_loader(local, split, train=False), resolve_device(str(config.get("device", "auto"))), str(config.get("precision", "fp32")))
    manifest = load_manifest(config["dataset"]["manifest"], root=config["dataset"].get("root"), strict_paths=False)
    candidates = manifest.loc[manifest["split"].astype(str).eq(split) & manifest["label"].astype(int).eq(0)].merge(predictions, on=["sample_id", "label"], how="inner")
    probability_columns = sorted((column for column in candidates.columns if column.startswith("p_")), key=lambda value: int(value[2:]))
    if len(probability_columns) < 2:
        raise ValueError("Hard-negative mining requires class probabilities including the safe class")
    candidates["mining_score"] = 1.0 - candidates["p_0"].astype(float)
    candidates["mining_label"] = candidates[probability_columns].to_numpy().argmax(1)
    candidates = candidates.loc[candidates["mining_score"] >= float(section.get("score_threshold", 0.70))]
    candidates = _temporal_nms(candidates, float(section.get("minimum_separation_s", 3.0)))
    per_source = int(section.get("max_per_source", 50))
    candidates = candidates.sort_values("mining_score", ascending=False).groupby("source_id", sort=False).head(per_source)
    candidates = candidates.head(int(section.get("max_candidates", 50000))).copy()
    candidates["is_hard_negative"] = True
    candidates["adjudication_status"] = str(section.get("initial_status", "pending_review"))
    keep = list(dict.fromkeys(list(manifest.columns) + ["mining_score", "mining_label", "prediction", "is_hard_negative", "adjudication_status"]))
    output = Path(section.get("output", "data/hard_negative_candidates.parquet"))
    save_manifest(candidates[[column for column in keep if column in candidates.columns]], output)
    return candidates


def build_hard_negative_replay(config: dict[str, Any]) -> pd.DataFrame:
    section = config["hard_negatives"]
    frame = load_manifest(section["manifest"], strict_paths=False)
    status_column = str(section.get("label_quality_column", "adjudication_status"))
    accepted = set(str(value) for value in section.get("accepted_statuses", ("confirmed_safe", "policy_exception")))
    if status_column not in frame.columns:
        raise ValueError(f"Hard-negative manifest lacks {status_column}")
    replay = frame.loc[frame[status_column].astype(str).isin(accepted)].copy()
    replay["label"] = 0
    replay["is_hard_negative"] = True
    output = Path(section.get("replay_manifest", "data/hard_negatives_replay.parquet"))
    save_manifest(replay, output)
    return replay


def _temporal_nms(frame: pd.DataFrame, separation_s: float) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for _, group in frame.sort_values("mining_score", ascending=False).groupby("source_id", sort=False):
        starts: list[float] = []
        for _, row in group.iterrows():
            start = float(row["start_s"])
            if all(abs(start - previous) >= separation_s for previous in starts):
                selected.append(row)
                starts.append(start)
    return pd.DataFrame(selected, columns=frame.columns)
