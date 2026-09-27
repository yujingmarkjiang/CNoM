from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.05))
    if lr <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay must be non-negative")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Optimizer requires at least one trainable parameter")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=tuple(config.get("betas", [0.9, 0.999])))
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=float(config.get("momentum", 0.9)), nesterov=True, weight_decay=weight_decay)
    if name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise RuntimeError("Install lion-pytorch to use the Lion optimizer") from exc
        return Lion(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: Optimizer, config: dict[str, Any], steps_per_epoch: int):
    name = str(config.get("scheduler", "cosine")).lower()
    epochs = int(config.get("epochs", 1))
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    total_steps = epochs * steps_per_epoch
    warmup_epochs = float(config.get("warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    warmup_steps = min(total_steps, int(warmup_epochs * steps_per_epoch))
    min_ratio = float(config.get("min_learning_rate", 0.0)) / max(float(config.get("learning_rate", 1e-4)), 1e-12)
    if name == "onecycle":
        return OneCycleLR(optimizer, max_lr=float(config["learning_rate"]), total_steps=total_steps, pct_start=max(0.01, warmup_steps / total_steps))
    if name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if name == "linear":
            return max(min_ratio, 1 - progress)
        if name == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        raise ValueError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, schedule)
