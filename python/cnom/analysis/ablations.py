from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cnom.analysis.benchmarks import run_benchmark_matrix
from cnom.config import deep_merge, set_dotpath
from cnom.runner import train_experiment
from cnom.utils import write_table


def run_ablation_matrix(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for variant in config["variants"]:
        name = str(variant["name"])
        local = deep_merge(config, {"output_dir": str(output_dir / name)})
        for key, value in variant.get("overrides", {}).items():
            set_dotpath(local, str(key), value)
        checkpoint = variant.get("sentinel_checkpoint")
        if bool(variant.get("train", checkpoint is None)):
            local["stage"] = str(variant.get("training_stage", "distillation"))
            train_experiment(local)
            checkpoint = str(Path(local["output_dir"]) / "best.safetensors")
        teacher_checkpoint = variant.get("teacher_checkpoint", local["teacher"].get("checkpoint"))
        benchmark = deep_merge(local, {
            "output_dir": str(Path(local["output_dir"]) / "benchmark"),
            "systems": [{"name": name, "mode": "cascade", "sentinel_checkpoint": checkpoint, "teacher_checkpoint": teacher_checkpoint}],
        })
        result = run_benchmark_matrix(benchmark)
        result.insert(0, "variant", name)
        rows.append(result)
    combined = pd.concat(rows, ignore_index=True)
    write_table(combined, output_dir / "ablation_matrix.csv")
    return combined
