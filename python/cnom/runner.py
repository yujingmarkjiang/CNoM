from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cnom.config import deep_merge
from cnom.data.datasets import FeatureBagDataset, VideoClipDataset, build_dataloader
from cnom.data.manifest import load_manifest
from cnom.inference import build_clip_loader, predict_dataset
from cnom.models.common import freeze_for_linear_probe
from cnom.models.preprocessing import resolve_video_preprocessing
from cnom.models.registry import build_model
from cnom.training.engine import Trainer
from cnom.training.losses import LossComputer
from cnom.training.temperature import TemperatureScaler
from cnom.utils import initialize_distributed, resolve_device, seed_everything, write_table

_FEATURE_MODELS = {"rtfm", "mgfn"}
_MODEL_KEYS = {
    "name", "backend", "pretrained_id", "pretrained_weights", "pretrained", "num_classes", "image_size",
    "num_frames", "mean", "std", "dropout", "gradient_checkpointing", "compile", "variant",
    "shared_backbone", "num_heads", "head_hidden_dim", "lse_temperature", "aggregation", "revision",
    "cache_dir", "local_files_only", "processor_local_files_only", "trust_remote_code", "repository",
    "source", "slowfast_alpha", "feature_dim", "hidden_dim", "num_segments", "num_stages", "top_k",
    "num_layers", "prompts", "freeze_clip", "checkpoint", "initial_checkpoint", "batch_size",
}


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    config = _with_auto_resume(config)
    training = config.get("training", {})
    distributed = initialize_distributed(str(config.get("device", "auto")), str(training.get("distributed_backend", "nccl")))
    seed_everything(int(config.get("seed", 2027)) + distributed.rank, bool(training.get("deterministic", False)))
    stage = str(config.get("stage", "teacher"))
    if stage == "distillation":
        model_cfg = _attach_grounding(dict(config["sentinel"]), config, "sentinel")
        teacher_cfg = _attach_grounding(dict(config["teacher"]), config, "teacher")
        model = build_model(model_cfg, checkpoint=_training_checkpoint(model_cfg), strict=False)
        teacher = build_model(teacher_cfg, checkpoint=teacher_cfg.get("checkpoint") or teacher_cfg.get("initial_checkpoint"), strict=False)
    else:
        model_cfg = _model_config(config)
        role = "sentinel" if str(model_cfg.get("name")) == "mobilenetv3_sentinel" else "teacher"
        model_cfg = _attach_grounding(model_cfg, config, role)
        model = build_model(model_cfg, checkpoint=_training_checkpoint(model_cfg), strict=False)
        teacher = None
    if bool(training.get("freeze_backbone", False)):
        freeze_for_linear_probe(model, tuple(training.get("trainable_patterns", ())))
    train_loader, validation_loader = _build_training_loaders(config, model_cfg, distributed)
    loss_cfg = dict(config.get("loss", {}))
    loss_cfg["label_smoothing"] = float(training.get("label_smoothing", 0.0))
    local = dict(config)
    local["device"] = str(distributed.device)
    trainer = Trainer(model, train_loader, validation_loader, local, LossComputer(loss_cfg), teacher=teacher, distributed=distributed)
    return trainer.fit()


def evaluate_experiment(config: dict[str, Any], split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_cfg = _model_config(config)
    role = "sentinel" if str(model_cfg.get("name")) == "mobilenetv3_sentinel" else "teacher"
    model_cfg = _attach_grounding(model_cfg, config, role)
    checkpoint = model_cfg.get("checkpoint") or str(Path(config.get("output_dir", "runs/default")) / "best.safetensors")
    model = build_model(model_cfg, checkpoint=checkpoint, strict=False)
    if str(model_cfg.get("name")) in _FEATURE_MODELS:
        dataset = FeatureBagDataset(
            config["dataset"]["manifest"], split, str(config["dataset"].get("feature_column", "feature_path")),
            root=config["dataset"].get("root"), num_segments=model_cfg.get("num_segments"),
        )
        loader = build_dataloader(dataset, int(config.get("training", {}).get("batch_size", 8)), int(config.get("workers", 4)), False)
    else:
        loader = build_clip_loader(config, split, train=False, model_config=model_cfg)
    predictions, metrics = predict_dataset(model, loader, resolve_device(str(config.get("device", "auto"))), str(config.get("precision", "fp32")))
    output_dir = Path(config.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(predictions, output_dir / f"{split}_clip_predictions.parquet")
    (output_dir / f"{split}_clip_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return predictions, metrics


def fit_temperature(config: dict[str, Any], logits_path: str | Path, output: str | Path) -> float:
    frame = pd.read_parquet(logits_path) if str(logits_path).endswith("parquet") else pd.read_csv(logits_path)
    columns = sorted([column for column in frame.columns if column.startswith("logit_")], key=lambda value: int(value.split("_")[-1]))
    if not columns:
        raise ValueError("Temperature fitting requires logit_0, logit_1, and subsequent logit columns")
    scaler = TemperatureScaler()
    temperature = scaler.fit(torch.tensor(frame[columns].to_numpy(), dtype=torch.float32), torch.tensor(frame["label"].to_numpy(), dtype=torch.long))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"temperature": temperature}, indent=2), encoding="utf-8")
    return temperature


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    top_level = {key: value for key, value in config.items() if key in _MODEL_KEYS}
    role = str(config.get("evaluation", {}).get("model_role", "")).lower()
    candidates: list[dict[str, Any]] = []
    if role in {"teacher", "sentinel", "model"} and isinstance(config.get(role), dict):
        candidates.append(dict(config[role]))
    if str(config.get("stage", "")).lower() == "distillation" and isinstance(config.get("sentinel"), dict):
        candidates.append(dict(config["sentinel"]))
    for key in ("model", "teacher", "sentinel"):
        if isinstance(config.get(key), dict):
            candidates.append(dict(config[key]))
    for candidate in candidates:
        if candidate.get("name"):
            return deep_merge(top_level, candidate)
    if not top_level.get("name"):
        raise ValueError("Configuration does not define a model name")
    return top_level



def _attach_grounding(model_cfg: dict[str, Any], config: dict[str, Any], role: str) -> dict[str, Any]:
    grounding = dict(config.get("grounding", {}))
    apply_to = set(grounding.get("apply_to", ("teacher",)))
    if grounding.get("enabled") and role in apply_to:
        model_cfg = dict(model_cfg)
        model_cfg["grounding"] = {**grounding, "enabled": True}
    return model_cfg

def _build_training_loaders(config: dict[str, Any], model_cfg: dict[str, Any], distributed):
    dataset_cfg = config["dataset"]
    manifest = _combined_manifest(dataset_cfg)
    manifest = _add_hard_negatives(manifest, config, dataset_cfg)
    training = config["training"]
    model_name = str(model_cfg.get("name"))
    if model_name in _FEATURE_MODELS:
        feature_column = str(dataset_cfg.get("feature_column", "feature_path"))
        train_dataset = FeatureBagDataset(
            manifest, str(dataset_cfg.get("train_split", "train")), feature_column, root=dataset_cfg.get("root"),
            num_segments=model_cfg.get("num_segments"),
        )
        validation_dataset = FeatureBagDataset(
            manifest, str(dataset_cfg.get("validation_split", "validation")), feature_column, root=dataset_cfg.get("root"),
            num_segments=model_cfg.get("num_segments"),
        )
    else:
        preprocessing = resolve_video_preprocessing(model_cfg, dataset_cfg)
        num_frames, image_size = preprocessing.num_frames, preprocessing.image_size
        grounding = bool(config.get("grounding", {}).get("enabled", False) and config.get("grounding", {}).get("precomputed", True))
        binary = bool(dataset_cfg.get("binary_labels", False))
        object_dim = int(config.get("grounding", {}).get("object_dim", 72))
        mean, std = preprocessing.mean, preprocessing.std
        augmentation = dict(dataset_cfg.get("augmentation", {}))
        common_video = {"num_frames": num_frames, "image_size": image_size, "decoder": str(dataset_cfg.get("decoder", "opencv")), "strict_paths": False, "include_object_features": grounding, "binary_labels": binary, "object_feature_dim": object_dim, "mean": mean, "std": std}
        if str(config.get("stage", "")) == "distillation":
            teacher_preprocessing = resolve_video_preprocessing(dict(config["teacher"]), dataset_cfg)
            if teacher_preprocessing.num_frames != num_frames:
                raise ValueError("Teacher and Sentinel must use the same num_frames during online distillation; use cached logits for different temporal inputs")
            common_video.update({"secondary_image_size": teacher_preprocessing.image_size, "secondary_mean": teacher_preprocessing.mean, "secondary_std": teacher_preprocessing.std})
        train_dataset = VideoClipDataset(manifest, str(dataset_cfg.get("train_split", "train")), sampling=str(dataset_cfg.get("sampling", "uniform")), train=True, horizontal_flip=float(augmentation.get("horizontal_flip", 0.5)), color_jitter=float(augmentation.get("color_jitter", 0.15)), random_erasing=float(augmentation.get("random_erasing", 0.0)), **common_video)
        validation_dataset = VideoClipDataset(manifest, str(dataset_cfg.get("validation_split", "validation")), sampling="uniform", train=False, horizontal_flip=0.0, color_jitter=0.0, random_erasing=0.0, **common_video)
    common = {"workers": int(config.get("workers", 4)), "distributed": distributed.enabled, "rank": distributed.rank, "world_size": distributed.world_size, "seed": int(config.get("seed", 2027))}
    train_loader = build_dataloader(train_dataset, int(training["batch_size"]), shuffle=True, class_balance=str(dataset_cfg.get("class_balance", "weighted")), drop_last=True, **common)
    validation_loader = build_dataloader(validation_dataset, int(training["batch_size"]), shuffle=False, class_balance="none", drop_last=False, **common)
    return train_loader, validation_loader


def _combined_manifest(dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    if not dataset_cfg.get("manifests"):
        return load_manifest(dataset_cfg["manifest"], root=dataset_cfg.get("root"), strict_paths=bool(dataset_cfg.get("strict_paths", True)))
    frames: list[pd.DataFrame] = []
    paths = list(dataset_cfg["manifests"])
    weights = list(dataset_cfg.get("dataset_weights", [1.0] * len(paths)))
    if len(weights) != len(paths):
        raise ValueError("dataset_weights must have one value per manifest")
    if any(float(weight) <= 0 for weight in weights):
        raise ValueError("dataset_weights must be positive")
    for index, (path, weight) in enumerate(zip(paths, weights, strict=True)):
        frame = load_manifest(path, root=dataset_cfg.get("root"), strict_paths=bool(dataset_cfg.get("strict_paths", True)))
        prefix = Path(path).stem or f"dataset_{index}"
        frame["sample_id"] = prefix + ":" + frame["sample_id"].astype(str)
        frame["source_id"] = prefix + ":" + frame["source_id"].astype(str)
        frame["sampling_weight"] = frame.get("sampling_weight", pd.Series(1.0, index=frame.index)).fillna(1.0).astype(float) * float(weight)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _add_hard_negatives(manifest: pd.DataFrame, config: dict[str, Any], dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    hard_cfg = config.get("hard_negatives", {})
    hard_manifest = hard_cfg.get("manifest")
    if not hard_manifest:
        return manifest
    hard = load_manifest(hard_manifest, root=dataset_cfg.get("root"), strict_paths=bool(dataset_cfg.get("strict_paths", True)))
    status_column, accepted = hard_cfg.get("label_quality_column"), set(hard_cfg.get("accepted_statuses", []))
    if status_column and status_column in hard.columns and accepted:
        hard = hard.loc[hard[status_column].isin(accepted)].copy()
    hard["is_hard_negative"], hard["label"], hard["split"] = True, 0, str(dataset_cfg.get("train_split", "train"))
    regular = manifest.loc[manifest["split"].astype(str) == str(dataset_cfg.get("train_split", "train"))]
    ratio = float(config.get("loss", {}).get("hard_negative_replay_ratio", config.get("hard_negatives", {}).get("replay_ratio", 1 / 3)))
    hard_weight = ratio / max(1e-9, 1 - ratio) * len(regular) / max(1, len(hard))
    manifest = manifest.copy()
    manifest["sampling_weight"] = manifest.get("sampling_weight", pd.Series(1.0, index=manifest.index)).fillna(1.0)
    hard["sampling_weight"] = hard_weight
    return pd.concat([manifest, hard], ignore_index=True)


def _with_auto_resume(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    enabled = bool(training.get("auto_resume", config.get("auto_resume", False)))
    if training.get("resume") or not enabled:
        return config
    candidate = Path(config.get("output_dir", "runs/default")) / "last.safetensors"
    if not candidate.is_file():
        return config
    local = dict(config)
    training["resume"] = str(candidate)
    local["training"] = training
    return local


def _training_checkpoint(model_config: dict[str, Any]) -> str | None:
    return model_config.get("initial_checkpoint") or model_config.get("checkpoint")
