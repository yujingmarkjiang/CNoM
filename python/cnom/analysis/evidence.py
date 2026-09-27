from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cnom.utils.hashing import sha256_file
from cnom.validation import validate_result_directory


def build_evidence_bundle(config: dict[str, Any]) -> dict[str, Any]:
    section = config["evidence"]
    root = Path(section["root"]).resolve()
    required = [Path(value) for value in section.get("required", ())]
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("Missing evidence artifacts: " + ", ".join(missing))
    validation = validate_result_directory(root) if bool(section.get("validate_results", True)) else None
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"evidence_manifest.json", "validation_report.json"}:
            records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    report = {"root": str(root), "files": records, "validation": validation, "passed": not missing and (validation is None or validation["passed"])}
    output = root / str(section.get("manifest_name", "evidence_manifest.json"))
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
