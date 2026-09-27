from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def safe_artifact_stem(value: Any, max_prefix: int = 72) -> str:
    text = str(value)
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")[:max_prefix] or "item"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"
