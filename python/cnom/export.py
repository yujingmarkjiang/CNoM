from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cnom.models.common import ModelOutput, count_parameters
from cnom.models.preprocessing import resolve_video_preprocessing
from cnom.models.registry import build_model
from cnom.utils.hashing import sha256_file
from cnom.utils.device import resolve_device


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor):
        output: ModelOutput = self.model(video)
        frame_logits = output.frame_logits if output.frame_logits is not None else output.logits.unsqueeze(1)
        return output.logits, frame_logits


def export_model(config: dict[str, Any], target: str, output: str | Path | None = None) -> Path:
    model_cfg = dict(config.get("sentinel", config.get("model", config.get("teacher", {}))))
    checkpoint = model_cfg.get("checkpoint") or Path(config.get("output_dir", "runs/default")) / "best.safetensors"
    model = build_model(model_cfg, checkpoint=str(checkpoint), strict=False).eval()
    device = resolve_device(str(config.get("device", "cpu")))
    model.to(device)
    wrapper = ExportWrapper(model).eval()
    preprocessing = resolve_video_preprocessing(model_cfg, config.get("dataset", {}))
    export_cfg = dict(config.get("export", {}))
    default_frames = 1 if str(model_cfg.get("name")) == "mobilenetv3_sentinel" else preprocessing.num_frames
    frames, size = int(export_cfg.get("num_frames", default_frames)), int(export_cfg.get("image_size", preprocessing.image_size))
    example = torch.randn(1, frames, 3, size, size, device=device)
    output_path = Path(output or f"artifacts/{model_cfg['name']}.{_suffix(target)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if target == "torchscript":
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(output_path))
    elif target == "onnx":
        torch.onnx.export(
            wrapper, example, str(output_path), input_names=["video"], output_names=["event_logits", "frame_logits"],
            dynamic_axes={"video": {0: "batch", 1: "frames"}, "event_logits": {0: "batch"}, "frame_logits": {0: "batch", 1: "frames"}},
            opset_version=18, do_constant_folding=True,
        )
    elif target == "torch_export":
        exported = torch.export.export(wrapper, (example,))
        torch.export.save(exported, str(output_path))
    else:
        raise ValueError(f"Unknown export target: {target}")
    metadata = {
        "model": model_cfg["name"], "target": target, "path": str(output_path), "sha256": sha256_file(output_path),
        "parameters": count_parameters(model), "input": [1, frames, 3, size, size], "labels": config.get("labels"),
        "preprocessing": {**preprocessing.to_dict(), "runtime_num_frames": frames, "runtime_image_size": size},
        "calibration": config.get("calibration", {}).get("selected"),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


def _suffix(target: str) -> str:
    return {"torchscript": "ts", "onnx": "onnx", "torch_export": "pt2"}[target]
