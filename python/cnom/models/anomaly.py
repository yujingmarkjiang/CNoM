from __future__ import annotations

import math

import torch
from torch import nn

from cnom.models.common import ModelOutput


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(dim, dim, 1), nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + residual)


class RTFM(nn.Module):
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512, top_k: int = 3, dropout: float = 0.7) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or top_k <= 0 or not 0 <= dropout < 1:
            raise ValueError("RTFM dimensions and top_k must be positive and dropout must be in [0, 1)")
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.temporal = nn.Sequential(TemporalBlock(hidden_dim, 1, dropout / 2), TemporalBlock(hidden_dim, 2, dropout / 2))
        self.score = nn.Linear(hidden_dim, 1)
        self.top_k = top_k

    def forward(self, features: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        x = self.temporal(self.project(features))
        segment_scores = self.score(x).squeeze(-1)
        magnitudes = x.norm(dim=-1)
        k = min(self.top_k, segment_scores.shape[1])
        top = torch.topk(magnitudes, k=k, dim=1).indices
        video_logits = torch.gather(segment_scores, 1, top).mean(1)
        logits = torch.stack([-video_logits, video_logits], dim=-1)
        return ModelOutput(logits=logits, features=x.mean(1), segment_scores=segment_scores)


class MagnitudeFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm = nn.LayerNorm(dim)
        self.magnitude = nn.Sequential(nn.Linear(1, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        weighted = x * (1 + self.magnitude(mag))
        attended, _ = self.attention(weighted, weighted, weighted, need_weights=False)
        x = self.norm(x + attended)
        return x + self.ffn(x)


class MGFN(nn.Module):
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512, num_stages: int = 3, num_heads: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or num_stages <= 0 or num_heads <= 0 or not 0 <= dropout < 1:
            raise ValueError("MGFN dimensions, stages, and heads must be positive and dropout must be in [0, 1)")
        if hidden_dim % num_heads:
            raise ValueError("MGFN hidden_dim must be divisible by num_heads")
        self.project = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([MagnitudeFusionBlock(hidden_dim, num_heads, dropout) for _ in range(num_stages)])
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor, object_features: torch.Tensor | None = None) -> ModelOutput:
        x = self.project(features)
        for block in self.blocks:
            x = block(x)
        segment_scores = self.score(x).squeeze(-1)
        video_score = torch.logsumexp(segment_scores, dim=1) - math.log(segment_scores.shape[1])
        logits = torch.stack([-video_score, video_score], dim=-1)
        return ModelOutput(logits=logits, features=x.mean(1), segment_scores=segment_scores)
