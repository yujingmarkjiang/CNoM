from __future__ import annotations

import torch
from torch import nn

from cnom.models.common import ModelOutput


class VadCLIP(nn.Module):
    def __init__(
        self,
        pretrained_id: str = "ViT-B-16",
        pretrained_weights: str = "laion2b_s34b_b88k",
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        prompts: dict[str, str] | None = None,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("Install open-clip-torch to use VadCLIP") from exc
        self.clip, _, _ = open_clip.create_model_and_transforms(pretrained_id, pretrained=pretrained_weights)
        self.freeze_clip = freeze_clip
        if freeze_clip:
            for parameter in self.clip.parameters():
                parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(pretrained_id)
        prompts = prompts or {"normal": "a normal video", "anomaly": "an unsafe video"}
        tokens = self.tokenizer([prompts["normal"], prompts["anomaly"]])
        self.register_buffer("prompt_tokens", tokens, persistent=False)
        width = int(getattr(self.clip.visual, "output_dim", hidden_dim))
        self.project = nn.Linear(width, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers)
        self.segment_head = nn.Linear(hidden_dim, 1)
        self.text_project = nn.Linear(width, hidden_dim, bias=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def forward(self, video: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        batch, frames, channels, height, width = video.shape
        images = video.reshape(batch * frames, channels, height, width)
        image_features = self.clip.encode_image(images, normalize=True).reshape(batch, frames, -1)
        x = self.temporal(self.project(image_features))
        segment_scores = self.segment_head(x).squeeze(-1)
        text = self.clip.encode_text(self.prompt_tokens, normalize=True)
        pooled = x.mean(1)
        prompt_logits = pooled @ self.text_project(text).transpose(0, 1)
        anomaly_score = segment_scores.topk(k=max(1, frames // 8), dim=1).values.mean(1)
        logits = prompt_logits + torch.stack([-anomaly_score, anomaly_score], dim=-1)
        return ModelOutput(logits=logits, features=pooled, segment_scores=segment_scores)
