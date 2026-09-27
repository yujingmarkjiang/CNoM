from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cnom.config import deep_merge
from cnom.data.manifest import load_manifest, save_manifest
from cnom.data.samplers import sample_low_label
from cnom.runner import evaluate_experiment, train_experiment
from cnom.utils import write_table


def run_low_label_study(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = load_manifest(config["dataset"]["manifest"], root=config["dataset"].get("root"), strict_paths=False)
    rows: list[dict[str, Any]] = []
    for budget in config["budgets"]:
        for seed in config["seeds"]:
            sampled = sample_low_label(base_manifest, int(budget), int(seed), str(config["sampling"].get("group_field", "source_id")), float(config["sampling"].get("preserve_safe_ratio", 1.0)))
            manifest_path = output_dir / "manifests" / f"budget_{budget}_seed_{seed}.parquet"
            save_manifest(sampled, manifest_path)
            for method in config["methods"]:
                run_dir = output_dir / f"budget_{budget}" / f"seed_{seed}" / str(method)
                local = _method_config(config, str(method), manifest_path, run_dir, int(seed))
                train_result = train_experiment(local)
                _, metrics = evaluate_experiment(local, str(local["dataset"].get("test_split", "test")))
                rows.append({
                    "labeled_events_per_class": int(budget), "seed": int(seed), "method": str(method),
                    "recall": metrics.get("macro_recall"), "precision": metrics.get("macro_precision"),
                    "macro_f1": metrics.get("macro_f1"), "accuracy": metrics.get("accuracy"),
                    "best_validation_metric": train_result.get("best_metric"),
                })
    raw = pd.DataFrame(rows)
    write_table(raw, output_dir / "low_label_runs.csv")
    summary = raw.groupby(["labeled_events_per_class", "method"]).agg(
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), precision_mean=("precision", "mean"),
        precision_std=("precision", "std"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    write_table(summary, output_dir / "low_label_summary.csv")
    return summary


def _method_config(config: dict[str, Any], method: str, manifest_path: Path, run_dir: Path, seed: int) -> dict[str, Any]:
    update: dict[str, Any] = {"seed": seed, "output_dir": str(run_dir), "dataset": {"manifest": str(manifest_path)}}
    if method == "mobilenetv3_supervised":
        update.update({"stage": "teacher", "model": {"name": "mobilenetv3_sentinel", "num_heads": 8, "num_classes": 9, "variant": "large", "pretrained": True, "shared_backbone": True}, "loss": {"distillation_weight": 0.0, "hard_negative_weight": 0.0}})
    elif method == "vivit_supervised":
        update.update({"stage": "teacher", "model": dict(config.get("teacher", {"name": "vivit", "pretrained_id": "google/vivit-b-16x2-kinetics400", "num_classes": 9})), "loss": {"distillation_weight": 0.0, "hard_negative_weight": 0.0}})
    elif method == "cnom_without_distillation":
        update.update({"stage": "distillation", "loss": {"distillation_weight": 0.0}})
    elif method == "cnom_full":
        update.update({"stage": "distillation"})
    else:
        raise ValueError(f"Unknown low-label method: {method}")
    return deep_merge(config, update)
