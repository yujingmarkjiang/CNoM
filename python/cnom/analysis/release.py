from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cnom.utils import read_table


def check_release(config: dict[str, Any]) -> dict[str, Any]:
    section = config["release"]
    current = _row(read_table(section["current_metrics"]), section.get("current_system"))
    baseline = _row(read_table(section["baseline_metrics"]), section.get("baseline_system")) if section.get("baseline_metrics") else None
    checks: list[dict[str, Any]] = []
    for metric, minimum in section.get("minimums", {}).items():
        value = float(current[metric])
        checks.append({"metric": metric, "rule": f">={minimum}", "value": value, "passed": value >= float(minimum)})
    for metric, maximum in section.get("maximums", {}).items():
        value = float(current[metric])
        checks.append({"metric": metric, "rule": f"<={maximum}", "value": value, "passed": value <= float(maximum)})
    if baseline is not None:
        for metric, allowed_drop in section.get("maximum_relative_drops", {}).items():
            value, reference = float(current[metric]), float(baseline[metric])
            drop = (reference - value) / abs(reference) if reference else 0.0
            checks.append({"metric": metric, "rule": f"relative_drop<={allowed_drop}", "value": value, "baseline": reference, "relative_drop": drop, "passed": drop <= float(allowed_drop)})
    passed = all(check["passed"] for check in checks)
    report = {"passed": passed, "decision": "promote" if passed else "block", "checks": checks}
    output = Path(section.get("output", Path(config.get("output_dir", "runs/default")) / "release_check.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed and bool(section.get("fail_process", True)):
        raise RuntimeError(f"Release blocked; see {output}")
    return report


def _row(frame, system: str | None):
    if system and "system" in frame.columns:
        selected = frame.loc[frame["system"].astype(str).eq(str(system))]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one metrics row for system={system}")
        return selected.iloc[0]
    if len(frame) != 1:
        raise ValueError("Metrics file must contain one row or a system selector")
    return frame.iloc[0]
