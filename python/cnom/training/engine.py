from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from cnom.metrics.classification import classification_metrics
from cnom.metrics.statistics import wilson_interval
from cnom.models.common import load_weights, unwrap_model
from cnom.training.checkpoint import load_training_state, save_checkpoint
from cnom.training.losses import LossComputer
from cnom.training.optim import build_optimizer, build_scheduler
from cnom.utils import DistributedContext, all_gather_objects, autocast_dtype, barrier

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -math.inf
    epochs_without_improvement: int = 0


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        missing = set(self.shadow).difference(state)
        if missing:
            self.shadow = {name: parameter.detach().clone() for name, parameter in parameters.items() if parameter.requires_grad}
            return
        self.shadow = {name: state[name].to(parameters[name].device).clone() for name in self.shadow}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        validation_loader,
        config: dict[str, Any],
        loss_computer: LossComputer,
        teacher: nn.Module | None = None,
        distributed: DistributedContext | None = None,
    ) -> None:
        self.config, self.train_config = config, config["training"]
        epochs = int(self.train_config.get("epochs", 0))
        accumulation = int(self.train_config.get("gradient_accumulation", 1))
        monitor_mode = str(self.train_config.get("monitor_mode", "max"))
        ema_decay = float(self.train_config.get("ema_decay", 0))
        if epochs <= 0 or accumulation <= 0:
            raise ValueError("training.epochs and training.gradient_accumulation must be positive")
        if monitor_mode not in {"min", "max"}:
            raise ValueError("training.monitor_mode must be min or max")
        if ema_decay and not 0 < ema_decay < 1:
            raise ValueError("training.ema_decay must be in (0, 1)")
        self.output_dir = Path(config.get("output_dir", "runs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.distributed = distributed or DistributedContext(False, 0, 0, 1, torch.device(str(config.get("device", "cpu"))))
        self.device = self.distributed.device
        model = model.to(self.device)
        if bool(self.train_config.get("compile", config.get("model", {}).get("compile", False))) and hasattr(torch, "compile"):
            model = torch.compile(model, mode=str(self.train_config.get("compile_mode", "default")))
        if self.distributed.enabled:
            model = DistributedDataParallel(model, device_ids=[self.distributed.local_rank] if self.device.type == "cuda" else None, find_unused_parameters=bool(self.train_config.get("find_unused_parameters", False)))
        self.model = model
        self.teacher = teacher.to(self.device).eval() if teacher is not None else None
        if self.teacher is not None:
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.loss_computer = loss_computer
        self.optimizer = build_optimizer(self.model, self.train_config)
        steps = math.ceil(len(train_loader) / int(self.train_config.get("gradient_accumulation", 1)))
        self.scheduler = build_scheduler(self.optimizer, self.train_config, steps)
        self.precision = str(config.get("precision", "fp32"))
        self.autocast_type = autocast_dtype(self.precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16" and self.device.type == "cuda")
        initial_best = math.inf if str(self.train_config.get("monitor_mode", "max")) == "min" else -math.inf
        self.state = TrainerState(best_metric=initial_best)
        self.ema = ExponentialMovingAverage(self._unwrapped_model(), float(self.train_config.get("ema_decay", 0))) if self.train_config.get("ema_decay") else None
        self.history: list[dict[str, float]] = []
        self._resume_if_configured()

    def fit(self) -> dict[str, Any]:
        fit_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        epochs = int(self.train_config["epochs"])
        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch
            self._set_sampler_epoch(epoch)
            train_metrics = self._run_epoch(training=True)
            use_ema = bool(self.ema and self.train_config.get("validation_use_ema", True))
            if use_ema:
                self.ema.apply(self._unwrapped_model())
            validation_metrics = self._run_epoch(training=False)
            if use_ema:
                self.ema.restore(self._unwrapped_model())
            metrics = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}}
            self.history.append(metrics)
            improved = self._is_improved(metrics)
            if improved:
                self.state.epochs_without_improvement = 0
                if self.ema and bool(self.train_config.get("save_ema_best", True)):
                    self.ema.apply(self._unwrapped_model())
                    self._save("best.safetensors")
                    self.ema.restore(self._unwrapped_model())
                else:
                    self._save("best.safetensors")
            else:
                self.state.epochs_without_improvement += 1
            if (epoch + 1) % int(self.train_config.get("save_every_epochs", 1)) == 0:
                self._save("last.safetensors")
            if self.distributed.is_main:
                self._write_history()
                LOGGER.info("epoch=%d train_loss=%.5f validation_macro_f1=%.5f validation_unsafe_recall=%.5f", epoch, train_metrics["loss"], validation_metrics["macro_f1"], validation_metrics["unsafe_recall"])
            if self.state.epochs_without_improvement >= int(self.train_config.get("early_stopping_patience", epochs + 1)):
                break
        barrier()
        wall_seconds = time.perf_counter() - fit_started
        peak_memory = int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0
        resources = {"wall_seconds": wall_seconds, "world_size": self.distributed.world_size, "accelerator_hours": wall_seconds * self.distributed.world_size / 3600, "peak_device_memory_bytes": peak_memory, "epochs_completed": len(self.history), "global_steps": self.state.global_step}
        if self.distributed.is_main:
            (self.output_dir / "training_resources.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")
        return {"best_metric": self.state.best_metric, "history": self.history, "resources": resources} if self.distributed.is_main else {"best_metric": self.state.best_metric, "history": [], "resources": resources}

    def _run_epoch(self, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.validation_loader
        self.model.train(training)
        losses: list[float] = []
        labels_all: list[np.ndarray] = []
        predictions_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []
        accumulation = int(self.train_config.get("gradient_accumulation", 1))
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        for step, batch in enumerate(loader):
            labels = batch["label"].to(self.device, non_blocking=True)
            inputs = batch["features" if "features" in batch else "video"].to(self.device, non_blocking=True)
            object_features = batch.get("object_features")
            if object_features is not None:
                object_features = object_features.to(self.device, non_blocking=True)
            hard_negative_mask = batch.get("is_hard_negative")
            if hard_negative_mask is not None:
                hard_negative_mask = hard_negative_mask.to(self.device, non_blocking=True).bool()
            update_now = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            sync_context = contextlib.nullcontext() if not training or update_now or not hasattr(self.model, "no_sync") else self.model.no_sync()
            autocast = torch.autocast(device_type=self.device.type, dtype=self.autocast_type, enabled=self.autocast_type is not None and self.device.type != "cpu")
            with sync_context, torch.set_grad_enabled(training), autocast:
                output = self.model(inputs, object_features=object_features)
                with torch.no_grad():
                    teacher_inputs = batch.get("teacher_video")
                    teacher_inputs = teacher_inputs.to(self.device, non_blocking=True) if teacher_inputs is not None else inputs
                    teacher_output = self.teacher(teacher_inputs, object_features=object_features) if self.teacher is not None else None
                loss_result = self.loss_computer(output, labels, teacher_output, hard_negative_mask)
                remainder = len(loader) % accumulation
                divisor = remainder if remainder and step >= len(loader) - remainder else accumulation
                loss = loss_result.total / divisor
            if training:
                self.scaler.scale(loss).backward()
                if update_now:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_config.get("max_grad_norm", 1.0)))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.state.global_step += 1
                    if self.ema:
                        self.ema.update(self._unwrapped_model())
            probabilities = output.logits.softmax(-1)
            losses.append(float(loss_result.total.detach()))
            labels_all.append(labels.detach().cpu().numpy())
            predictions_all.append(probabilities.argmax(-1).detach().cpu().numpy())
            probabilities_all.append(probabilities.detach().cpu().numpy())
        local = {
            "losses": losses,
            "labels": np.concatenate(labels_all) if labels_all else np.empty(0, dtype=int),
            "predictions": np.concatenate(predictions_all) if predictions_all else np.empty(0, dtype=int),
            "probabilities": np.concatenate(probabilities_all) if probabilities_all else np.empty((0, 1)),
            "elapsed": time.perf_counter() - started,
        }
        gathered = all_gather_objects(local)
        y_true = np.concatenate([value["labels"] for value in gathered])
        y_pred = np.concatenate([value["predictions"] for value in gathered])
        probabilities = np.concatenate([value["probabilities"] for value in gathered])
        losses = [item for value in gathered for item in value["losses"]]
        if len(y_true) == 0 or not losses:
            raise ValueError("Training or validation loader produced no samples")
        metrics = classification_metrics(y_true, y_pred, probabilities, probabilities.shape[1])
        unsafe = y_true != 0
        unsafe_tp = int(np.sum(unsafe & (y_pred != 0)))
        unsafe_trials = int(np.sum(unsafe))
        unsafe_recall = unsafe_tp / unsafe_trials if unsafe_trials else 0.0
        recall_lower = wilson_interval(unsafe_tp, unsafe_trials)[0] if unsafe_trials else 0.0
        throughput = len(y_true) / max(1e-6, max(value["elapsed"] for value in gathered))
        return {"loss": float(np.mean(losses)), "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"], "unsafe_recall": unsafe_recall, "recall_lower_bound": recall_lower, "samples_per_second": throughput}

    def _is_improved(self, metrics: dict[str, float]) -> bool:
        monitor = str(self.train_config.get("monitor", "validation_macro_f1"))
        value = float(metrics.get(monitor, metrics.get("validation_macro_f1", -math.inf)))
        mode = str(self.train_config.get("monitor_mode", "max"))
        improved = value > self.state.best_metric if mode == "max" else value < self.state.best_metric
        if improved:
            self.state.best_metric = value
        return improved

    def _save(self, name: str) -> None:
        if self.distributed.is_main:
            save_checkpoint(
                self.model, self.output_dir / name, self.optimizer, self.scheduler, self.scaler, self.state, self.config,
                ema_state=self.ema.state_dict() if self.ema else None,
            )
        barrier()

    def _resume_if_configured(self) -> None:
        resume = self.train_config.get("resume")
        if not resume:
            return
        load_weights(self._unwrapped_model(), resume, strict=True)
        payload = load_training_state(resume, self.optimizer, self.scheduler, self.scaler)
        state = payload.get("state") or {}
        defaults = TrainerState(best_metric=math.inf if str(self.train_config.get("monitor_mode", "max")) == "min" else -math.inf)
        values = {key: state.get(key, getattr(defaults, key)) for key in TrainerState.__dataclass_fields__}
        self.state = TrainerState(**values)
        if self.ema:
            self.ema.load_state_dict(payload.get("ema") or {}, self._unwrapped_model())
        self.state.epoch += 1

    def _set_sampler_epoch(self, epoch: int) -> None:
        for loader in (self.train_loader, self.validation_loader):
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def _unwrapped_model(self) -> nn.Module:
        return unwrap_model(self.model)

    def _write_history(self) -> None:
        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
