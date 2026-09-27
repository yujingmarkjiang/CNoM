from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from cnom.models.common import unwrap_model
from cnom.utils.hashing import sha256_file


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    state: Any = None,
    config: dict | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = unwrap_model(model).state_dict()
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file({key: value.detach().cpu().contiguous() for key, value in model_state.items()}, str(path))
    else:
        torch.save(model_state, path)
    training_path = path.with_suffix(".training.pt")
    payload = {
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "state": asdict(state) if is_dataclass(state) else state,
        "config": json.loads(json.dumps(config, default=str)) if config is not None else None,
        "ema": {key: value.detach().cpu() for key, value in ema_state.items()} if ema_state else None,
    }
    torch.save(payload, training_path)
    metadata = {"weights": str(path), "weights_sha256": sha256_file(path), "training_state": str(training_path)}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_training_state(path: str | Path, optimizer=None, scheduler=None, scaler=None) -> dict[str, Any]:
    training_path = Path(path).with_suffix(".training.pt")
    if not training_path.is_file():
        return {}
    payload = torch.load(training_path, map_location="cpu", weights_only=True)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return payload
