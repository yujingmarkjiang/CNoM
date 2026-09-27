from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

from cnom.models.common import ModelOutput, replace_last_linear


class HuggingFaceVideoClassifier(nn.Module):
    MODEL_CLASSES = {
        "vivit": "VivitForVideoClassification",
        "videomae": "VideoMAEForVideoClassification",
        "timesformer": "TimesformerForVideoClassification",
    }

    def __init__(
        self, name: str, pretrained_id: str, num_classes: int, pretrained: bool = True,
        gradient_checkpointing: bool = False, dropout: float = 0.2, num_frames: int | None = None,
        image_size: int | None = None, cache_dir: str | None = None, revision: str | None = None,
        local_files_only: bool = False, trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError("Install the training extra to use Transformers video models") from exc
        class_name = self.MODEL_CLASSES[name]
        model_class = getattr(transformers, class_name)
        hub_args = {"cache_dir": cache_dir, "revision": revision, "local_files_only": local_files_only, "trust_remote_code": trust_remote_code}
        hub_args = {key: value for key, value in hub_args.items() if value is not None}
        model_config = transformers.AutoConfig.from_pretrained(pretrained_id, num_labels=num_classes, **hub_args)
        for attribute in ("hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "dropout_rate"):
            if hasattr(model_config, attribute):
                setattr(model_config, attribute, dropout)
        if num_frames is not None and hasattr(model_config, "num_frames"):
            model_config.num_frames = num_frames
        if image_size is not None and hasattr(model_config, "image_size"):
            model_config.image_size = image_size
        self.model = model_class.from_pretrained(pretrained_id, config=model_config, ignore_mismatched_sizes=True, **hub_args) if pretrained else model_class(model_config)
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.name = name

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        output = self.model(pixel_values=video, output_hidden_states=True, return_dict=True)
        features = None
        if output.hidden_states:
            hidden = output.hidden_states[-1]
            features = hidden.mean(1)
        return ModelOutput(logits=output.logits, features=features)


class TorchvisionSwin3D(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.model = swin3d_t(weights=weights, dropout=dropout)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features
        self._features: torch.Tensor | None = None
        self.model.head.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, module, inputs) -> None:
        self._features = inputs[0]

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        logits = self.model(video.permute(0, 2, 1, 3, 4))
        return ModelOutput(logits=logits, features=self._features)


class TorchHubVideoClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        repository: str,
        num_classes: int,
        pretrained: bool = True,
        source: str = "github",
        slowfast_alpha: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = torch.hub.load(repository, model_name, pretrained=pretrained, source=source)
        self.feature_dim = replace_last_linear(self.model, num_classes)
        if dropout > 0:
            linears = [(parent, name, child) for parent in self.model.modules() for name, child in parent.named_children() if isinstance(child, nn.Linear)]
            parent, name, layer = linears[-1]
            setattr(parent, name, nn.Sequential(nn.Dropout(dropout), layer))
        self.model_name = model_name
        self.slowfast_alpha = slowfast_alpha
        self._features: torch.Tensor | None = None
        linears = [module for module in self.model.modules() if isinstance(module, nn.Linear)]
        linears[-1].register_forward_pre_hook(self._capture_features)

    def _capture_features(self, module, inputs) -> None:
        self._features = inputs[0]

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        tensor = video.permute(0, 2, 1, 3, 4)
        if "slowfast" in self.model_name:
            inputs: Any = [tensor[:, :, :: self.slowfast_alpha], tensor]
        else:
            inputs = tensor
        logits = self.model(inputs)
        features = self._features.flatten(1) if self._features is not None else None
        return ModelOutput(logits=logits, features=features)
