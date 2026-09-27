from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor | None = None
    frame_logits: torch.Tensor | None = None
    segment_scores: torch.Tensor | None = None
    extra: dict[str, Any] | None = None


def replace_last_linear(module: nn.Module, out_features: int) -> int:
    candidates: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in module.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                candidates.append((parent, name, child))
    if not candidates:
        raise ValueError(f"No linear layer found in {module.__class__.__name__}")
    parent, name, layer = candidates[-1]
    setattr(parent, name, nn.Linear(layer.in_features, out_features, bias=layer.bias is not None))
    return layer.in_features


def load_weights(model: nn.Module, checkpoint: str | Path | None, strict: bool = True) -> None:
    if not checkpoint:
        return
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Install safetensors to load this checkpoint") from exc
        state = load_file(str(path))
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Checkpoint {path} is not a weights-only PyTorch checkpoint") from exc
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError(f"Checkpoint {path} does not contain a tensor state dictionary")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if not strict:
        target = model.state_dict()
        candidates = [state, {f"base.{key}": value for key, value in state.items()}, {key.removeprefix("base."): value for key, value in state.items()}]
        state = max(candidates, key=lambda candidate: sum(key in target and target[key].shape == value.shape for key, value in candidate.items()))
        state = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
        if not state:
            raise ValueError(f"Checkpoint {path} has no compatible parameters for {model.__class__.__name__}")
    model.load_state_dict(state, strict=strict)



def unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while hasattr(current, "module"):
        current = current.module
    return getattr(current, "_orig_mod", current)

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)


def freeze_for_linear_probe(model: nn.Module, trainable_patterns: tuple[str, ...] = ()) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[str] = []
    if trainable_patterns:
        for name, parameter in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                parameter.requires_grad_(True)
                selected.append(name)
    else:
        linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        if not linears:
            raise ValueError("Linear-probe mode requires at least one linear layer")
        last_name, last_layer = linears[-1]
        for parameter in last_layer.parameters():
            parameter.requires_grad_(True)
        selected.extend(f"{last_name}.{name}" for name, _ in last_layer.named_parameters())
    if not selected:
        raise ValueError("No parameters matched the configured linear-probe patterns")
    return selected
