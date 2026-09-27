from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cnom.metrics.statistics import bootstrap_ci
from cnom.utils import read_table, write_table


def analyze_operational_outcomes(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["input"]).copy()
    time_column = str(config["time_column"])
    frame[time_column] = pd.to_datetime(frame[time_column])
    intervention = pd.Timestamp(config["intervention_date"])
    frame = frame.sort_values(time_column)
    frame["time_index"] = (frame[time_column] - frame[time_column].min()).dt.total_seconds() / 86400
    frame["deployed"] = (frame[time_column] >= intervention).astype(int)
    frame["post_time"] = np.where(frame["deployed"].eq(1), (frame[time_column] - intervention).dt.total_seconds() / 86400, 0.0)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for outcome in config["outcomes"]:
        name = str(outcome["name"])
        subset = frame.dropna(subset=[name]).copy()
        baseline = subset.loc[subset["deployed"] == 0, name].to_numpy(dtype=float)
        deployed = subset.loc[subset["deployed"] == 1, name].to_numpy(dtype=float)
        baseline_value = float(np.median(baseline) if outcome.get("family") == "quantile" else np.mean(baseline))
        deployed_value = float(np.median(deployed) if outcome.get("family") == "quantile" else np.mean(deployed))
        absolute = deployed_value - baseline_value
        relative = 100 * absolute / baseline_value if baseline_value else float("nan")
        model_result = _fit_model(subset, name, outcome, config)
        ci = _relative_change_ci(subset, name, outcome, config)
        rows.append({
            "outcome": name, "baseline_value": baseline_value, "deployed_value": deployed_value,
            "absolute_change": absolute, "relative_change_percent": relative, "ci_low": ci[0], "ci_high": ci[1],
            "n_baseline": len(baseline), "n_deployed": len(deployed), "p_value": model_result.get("p_value"),
            "effect": model_result.get("effect"), "standard_error": model_result.get("standard_error"),
            "family": outcome.get("family"), "primary": bool(outcome.get("primary", False)),
            "missing_percent": 100 * frame[name].isna().mean(),
        })
        models[name] = model_result
    result = pd.DataFrame(rows)
    write_table(result, output_dir / "operational_outcomes.csv")
    (output_dir / "model_details.json").write_text(json.dumps(models, indent=2, default=str), encoding="utf-8")
    _sensitivity_analyses(frame, config, output_dir)
    return result


def _fit_model(frame: pd.DataFrame, outcome_name: str, outcome: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("Install statsmodels to run operational outcome models") from exc
    covariates = [value for value in config.get("covariates", []) if value in frame.columns]
    design = str(config.get("design", "interrupted_time_series"))
    if design == "difference_in_differences":
        treatment = str(config.get("treatment_column", "treated"))
        formula = f"{outcome_name} ~ deployed * {treatment} + time_index" + "".join(f" + {value}" for value in covariates)
        effect_name = f"deployed:{treatment}"
    else:
        formula = f"{outcome_name} ~ time_index + deployed + post_time" + "".join(f" + {value}" for value in covariates)
        effect_name = "deployed"
    family = str(outcome.get("family", "gaussian"))
    if family == "quantile":
        result = smf.quantreg(formula, frame).fit(q=float(outcome.get("quantile", 0.5)), max_iter=5000)
    else:
        family_object = {
            "gaussian": sm.families.Gaussian(), "gamma": sm.families.Gamma(sm.families.links.Log()),
            "negative_binomial": sm.families.NegativeBinomial(), "binomial": sm.families.Binomial(),
        }.get(family)
        if family_object is None:
            raise ValueError(f"Unknown outcome family: {family}")
        fit_kwargs: dict[str, Any] = {}
        offset = outcome.get("offset")
        if offset and offset in frame.columns:
            fit_kwargs["offset"] = np.log(frame[offset].clip(lower=1e-9))
        model = smf.glm(formula, frame, family=family_object, **fit_kwargs)
        cluster_column = str(config.get("cluster_column", ""))
        result = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster_column]}) if cluster_column in frame.columns else model.fit()
    return {
        "formula": formula, "effect_name": effect_name, "effect": float(result.params.get(effect_name, np.nan)),
        "standard_error": float(result.bse.get(effect_name, np.nan)), "p_value": float(result.pvalues.get(effect_name, np.nan)),
        "confidence_interval": result.conf_int().loc[effect_name].tolist() if effect_name in result.params else None,
        "nobs": float(result.nobs), "summary": result.summary().as_text(),
    }


def _relative_change_ci(frame: pd.DataFrame, outcome_name: str, outcome: dict[str, Any], config: dict[str, Any]) -> tuple[float, float]:
    baseline = frame.loc[frame["deployed"] == 0]
    deployed = frame.loc[frame["deployed"] == 1]
    if baseline.empty or deployed.empty:
        return float("nan"), float("nan")
    repetitions = int(config.get("bootstrap_repetitions", 5000))
    cluster_column = str(config.get("cluster_column", ""))
    statistic = np.median if str(outcome.get("family")) == "quantile" else np.mean
    rng = np.random.default_rng(int(config.get("seed", 2027)))
    values = np.empty(repetitions)
    for index in range(repetitions):
        left = _bootstrap_sample(baseline, outcome_name, cluster_column, rng)
        right = _bootstrap_sample(deployed, outcome_name, cluster_column, rng)
        left_value, right_value = float(statistic(left)), float(statistic(right))
        values[index] = 100 * (right_value - left_value) / left_value if left_value else np.nan
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def _bootstrap_sample(frame: pd.DataFrame, outcome_name: str, cluster_column: str, rng: np.random.Generator) -> np.ndarray:
    if cluster_column and cluster_column in frame.columns:
        groups = frame[cluster_column].dropna().unique()
        sampled = rng.choice(groups, size=len(groups), replace=True)
        values = [frame.loc[frame[cluster_column] == group, outcome_name].to_numpy(dtype=float) for group in sampled]
        return np.concatenate(values) if values else frame[outcome_name].to_numpy(dtype=float)
    values = frame[outcome_name].to_numpy(dtype=float)
    return values[rng.integers(0, len(values), len(values))]


def _sensitivity_analyses(frame: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    time_column = str(config["time_column"])
    for window in config.get("exclude_windows", []):
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        subset = frame.loc[~frame[time_column].between(start, end)].copy()
        for outcome in config["outcomes"]:
            name = str(outcome["name"])
            if subset[name].notna().sum() == 0:
                continue
            model = _fit_model(subset.dropna(subset=[name]), name, outcome, config)
            rows.append({"exclusion": window["name"], "outcome": name, "effect": model["effect"], "p_value": model["p_value"], "nobs": model["nobs"]})
    write_table(pd.DataFrame(rows), output_dir / "sensitivity_analyses.csv")
