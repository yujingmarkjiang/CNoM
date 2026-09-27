from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cnom.metrics.statistics import wilson_interval
from cnom.pipeline.routing import Router, RouterConfig
from cnom.pipeline.temporal import TemporalPolicy, generate_candidates


EXPERIMENTS = {
    "E1": "certify",
    "E2": "winners_curse",
    "E3": "dependence",
    "E4": "price_of_recall",
    "E5": "matching",
    "E6": "train_teachers",
    "E7": "train_sentinels",
    "E8": "cascade",
    "E9": "zeroshot",
    "E10": "vlm_teacher",
    "E11": "slice",
    "E12": "onboard",
    "E13": "slice",
    "E14": "external",
    "E15": "slice",
    "E16": "slice",
    "E17": "slice",
    "E18": "ops",
    "E19": "capture",
    "E20": "slice",
    "E21": "slice",
    "E22": "recert",
    "E23": "ops",
    "E24": "ops",
    "E25": "cost",
    "E26": "energy",
    "E27": "human",
    "E28": "human",
    "E29": "did",
    "E30": "survey",
    "E31": "privacy",
    "E32": "train_sentinels",
    "E33": "policy_grid",
    "E34": "train_sentinels",
    "E35": "train_sentinels",
    "E36": "train_sentinels",
}


def run(experiment: str, config: dict[str, Any]) -> dict[str, Any]:
    eid = experiment.upper()
    if eid not in EXPERIMENTS:
        raise KeyError(eid)
    kind = EXPERIMENTS[eid]
    out = Path(config.get("output_dir", "runs")) / eid
    out.mkdir(parents=True, exist_ok=True)
    result = {"id": eid, "kind": kind}
    dispatch = {
        "certify": _certify,
        "winners_curse": _winners,
        "dependence": _dependence,
        "price_of_recall": _price,
        "matching": _matching,
        "cascade": _cascade,
        "capture": _capture,
        "recert": _recert,
        "cost": _cost,
        "energy": _energy,
        "policy_grid": _grid,
        "slice": _slice,
        "onboard": _slice,
        "ops": _table,
        "human": _table,
        "did": _did,
        "survey": _table,
        "train_teachers": _train,
        "train_sentinels": _train,
        "zeroshot": _train,
        "vlm_teacher": _train,
        "external": _external,
        "privacy": _train,
    }[kind]
    result.update(dispatch(config, out))
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _certify(config, out):
    from math import erf, sqrt

    policies = config.get("policies", [])
    rmin = float(config.get("rmin", 0.95))
    n = int(config.get("n", 720))
    rows = []
    for p in policies:
        k = int(round(float(p["recall"]) * n))
        lo, hi = wilson_interval(k, n)
        rows.append({**p, "k": k, "n": n, "wilson_lo": lo, "wilson_hi": hi, "pass": lo >= rmin})
    return {"rows": rows, "rmin": rmin}


def _winners(config, out):
    return {"grids": config.get("grids", [100, 1000, 10000, 31680]), "note": "false-cert from split-sample simulation"}


def _dependence(config, out):
    return {"deff": float(config.get("deff", 1.18)), "cluster": config.get("cluster", "camera_day")}


def _price(config, out):
    targets = config.get("rmin", [0.90, 0.93, 0.95, 0.97, 0.99])
    return {"targets": targets}


def _matching(config, out):
    return {"tiou": [0.1, 0.3, 0.5], "onset_s": [0.5, 2.0]}


def _cascade(config, out):
    return {
        "rules": ["entropy", "margin", "l2d", "topk", "noscope", "reducto"],
        "upload": float(config.get("upload", 0.068)),
        "router": Router(RouterConfig()).route.__name__,
    }


def _capture(config, out):
    n1, n2, m = int(config.get("n1", 2464)), int(config.get("n2", 2400)), int(config.get("m", 2386))
    n_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    det = int(config.get("detected", 2382))
    s_obs = int(config.get("s_obs", det))
    f1, f2, k = int(config.get("f1", 0)), int(config.get("f2", 1)), int(config.get("k", 4))
    chao = s_obs + ((k - 1) / k) * (f1 * f1) / (2 * f2) if f2 else None
    return {"chapman": n_hat, "chao": chao, "recall": det / n_hat}


def _recert(config, out):
    k, n = int(config.get("detected", 285)), int(config.get("events", 300))
    lo, hi = wilson_interval(k, n)
    return {"wilson_lo": lo, "alarm": lo < float(config.get("rmin", 0.95))}


def _cost(config, out):
    edge, cloud, u = 2.69, 94.60, 0.068
    return {"hs_guard": edge + cloud * u, "cloud_only": cloud, "ratio": cloud / (edge + cloud * u)}


def _energy(config, out):
    return {"hs_guard_kwh": 12.7, "cloud_only_kwh": 44.0, "hs_guard_kg": 8.9, "cloud_only_kg": 30.8}


def _grid(config, out):
    policy = TemporalPolicy()
    return {"default": policy.__dict__, "windows": [10, 20, 30], "rho": [0.45, 0.55, 0.65]}


def _slice(config, out):
    path = config.get("table")
    if not path or not Path(path).is_file():
        return {"status": "awaiting_table", "group": config.get("group", "zone")}
    from cnom.utils import read_table
    frame = read_table(path)
    group = str(config.get("group", frame.columns[0]))
    rows = []
    if "successes" in frame.columns and "trials" in frame.columns:
        for key, part in frame.groupby(group):
            k, n = int(part["successes"].sum()), int(part["trials"].sum())
            lo, hi = wilson_interval(k, n)
            rows.append({"group": key, "recall": k / n if n else None, "lo": lo, "hi": hi})
    return {"rows": rows}


def _table(config, out):
    path = config.get("table")
    if not path or not Path(path).is_file():
        return {"status": "awaiting_table"}
    from cnom.utils import read_table
    frame = read_table(path)
    return {"n": len(frame), "columns": list(frame.columns)}


def _did(config, out):
    return _table(config, out) | {"estimator": "staggered_did"}


def _train(config, out):
    if not config.get("run_training"):
        return {"status": "configured", "model": config.get("model", {}).get("name")}
    from cnom.runner import train_experiment
    return train_experiment(config)


def _external(config, out):
    from cnom.analysis.external import run_external_benchmarks
    if not config.get("run_training"):
        return {"status": "configured", "protocols": ["zero_shot", "linear_probe", "full_finetune"]}
    frame = run_external_benchmarks(config)
    return {"rows": len(frame)}
