from __future__ import annotations

from typing import Any

from torch import nn

from cnom.models.anomaly import MGFN, RTFM
from cnom.models.common import load_weights
from cnom.models.extra import AutoVideoClassifier, LinearProbe, QwenVLClassifier, TimmImageSentinel
from cnom.models.grounding import GroundedClassifier
from cnom.models.sentinel import MobileNetV3Sentinel
from cnom.models.vadclip import VadCLIP
from cnom.models.video_classifiers import HuggingFaceVideoClassifier, TorchHubVideoClassifier, TorchvisionSwin3D


def build_model(config: dict[str, Any], checkpoint: str | None = None, strict: bool = True) -> nn.Module:
    name = str(config["name"]).strip()
    num_classes = int(config.get("num_classes", 9))
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if name in {"rtfm", "mgfn", "vadclip"} and num_classes != 2:
        raise ValueError(f"{name} is a binary anomaly model and requires num_classes=2")
    if name == "mobilenetv3_sentinel":
        model: nn.Module = MobileNetV3Sentinel(
            num_classes=num_classes, num_heads=int(config.get("num_heads", num_classes - 1)), variant=str(config.get("variant", "large")),
            pretrained=bool(config.get("pretrained", True)), dropout=float(config.get("dropout", 0.2)),
            shared_backbone=bool(config.get("shared_backbone", True)), aggregation=str(config.get("aggregation", "max")),
            lse_temperature=float(config.get("lse_temperature", 8.0)), head_hidden_dim=int(config.get("head_hidden_dim", 0)),
        )
    elif name in {"vivit", "videomae", "timesformer"}:
        model = HuggingFaceVideoClassifier(
            name=name, pretrained_id=str(config["pretrained_id"]), num_classes=num_classes,
            pretrained=bool(config.get("pretrained", True)), gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
            dropout=float(config.get("dropout", 0.2)), num_frames=int(config.get("num_frames", 32)), image_size=int(config.get("image_size", 224)),
            cache_dir=config.get("cache_dir"), revision=config.get("revision"), local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    elif name == "video_swin_t":
        model = TorchvisionSwin3D(num_classes=num_classes, pretrained=bool(config.get("pretrained", True)), dropout=float(config.get("dropout", 0.2)))
    elif name in {"x3d_xs", "x3d_s", "x3d_m", "slowfast_r50"}:
        model = TorchHubVideoClassifier(
            model_name=name, repository=str(config.get("repository", "facebookresearch/pytorchvideo")), num_classes=num_classes,
            pretrained=bool(config.get("pretrained", True)), source=str(config.get("source", "github")),
            slowfast_alpha=int(config.get("slowfast_alpha", 4)), dropout=float(config.get("dropout", 0.0)),
        )
    elif name == "rtfm":
        feature_dim, hidden_dim, top_k = int(config.get("feature_dim", 2048)), int(config.get("hidden_dim", 512)), int(config.get("top_k", 3))
        if feature_dim <= 0 or hidden_dim <= 0 or top_k <= 0:
            raise ValueError("RTFM feature_dim, hidden_dim, and top_k must be positive")
        model = RTFM(feature_dim=feature_dim, hidden_dim=hidden_dim, top_k=top_k, dropout=float(config.get("dropout", 0.7)))
    elif name == "mgfn":
        feature_dim, hidden_dim = int(config.get("feature_dim", 2048)), int(config.get("hidden_dim", 512))
        num_stages, num_heads = int(config.get("num_stages", 3)), int(config.get("num_heads", 8))
        if feature_dim <= 0 or hidden_dim <= 0 or num_stages <= 0 or num_heads <= 0:
            raise ValueError("MGFN dimensions, stages, and heads must be positive")
        if hidden_dim % num_heads:
            raise ValueError("MGFN hidden_dim must be divisible by num_heads")
        model = MGFN(feature_dim=feature_dim, hidden_dim=hidden_dim, num_stages=num_stages, num_heads=num_heads, dropout=float(config.get("dropout", 0.3)))
    elif name == "vadclip":
        model = VadCLIP(
            pretrained_id=str(config.get("pretrained_id", "ViT-B-16")), pretrained_weights=str(config.get("pretrained_weights", "laion2b_s34b_b88k")),
            hidden_dim=int(config.get("hidden_dim", 512)), num_layers=int(config.get("num_layers", 4)),
            num_heads=int(config.get("num_heads", 8)), dropout=float(config.get("dropout", 0.2)), prompts=dict(config.get("prompts", {})),
            freeze_clip=bool(config.get("freeze_clip", True)),
        )
    elif name in {"videomae_v2", "internvideo2", "lavad"}:
        model = AutoVideoClassifier(
            pretrained_id=str(config["pretrained_id"]), num_classes=num_classes,
            pretrained=bool(config.get("pretrained", True)), cache_dir=config.get("cache_dir"),
            revision=config.get("revision"), local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=True,
        )
    elif name in {"vjepa2", "vjepa2_probe"}:
        model = LinearProbe(int(config.get("feature_dim", 1408)), num_classes)
    elif name in {"qwen3_vl_8b", "qwen3_vl_32b"}:
        model = QwenVLClassifier(str(config["pretrained_id"]), num_classes, cache_dir=config.get("cache_dir"), trust_remote_code=True)
    elif name in {"mobilenetv4_conv_s", "mobilenetv4_hybrid_m", "efficientvit_b1"}:
        arch = {
            "mobilenetv4_conv_s": "mobilenetv4_conv_small",
            "mobilenetv4_hybrid_m": "mobilenetv4_hybrid_medium",
            "efficientvit_b1": "efficientvit_b1",
        }[name]
        model = TimmImageSentinel(arch, num_classes, num_heads=int(config.get("num_heads", num_classes - 1)), pretrained=bool(config.get("pretrained", True)))
    else:
        raise ValueError(f"Unknown model: {name}")
    grounding = config.get("grounding")
    if grounding and grounding.get("enabled"):
        categories = tuple(grounding.get("categories", ()))
        if not categories and not grounding.get("precomputed"):
            raise ValueError("Grounding requires at least one COCO category")
        expected_dim = max(1, len(categories)) * 9 if categories else int(grounding.get("object_dim", 64))
        object_dim = int(grounding.get("object_dim", expected_dim))
        if categories and object_dim != expected_dim:
            raise ValueError(f"grounding.object_dim must be {expected_dim} for {len(categories)} categories")
        model = GroundedClassifier(model, num_classes, object_dim, int(grounding.get("hidden_dim", 64)))
    load_weights(model, checkpoint or config.get("checkpoint"), strict=strict)
    return model
