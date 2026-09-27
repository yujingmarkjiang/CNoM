from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(precision: str) -> torch.dtype | None:
    normalized = str(precision).lower()
    if normalized not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of fp32, fp16, or bf16")
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(normalized)
