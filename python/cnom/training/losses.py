from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cnom.models.common import ModelOutput


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.ndim > 2:
        features = features.flatten(1)
    features = F.normalize(features, dim=-1)
    similarity = features @ features.T / temperature
    mask = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    mask = mask & ~diagonal
    logits_mask = ~diagonal
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = mask.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return features.sum() * 0
    mean_log_prob = (log_prob * mask).sum(dim=1) / positives.clamp_min(1)
    return -mean_log_prob[valid].mean()


@dataclass
class LossResult:
    total: torch.Tensor
    supervised: torch.Tensor
    contrastive: torch.Tensor
    distillation: torch.Tensor
    false_negative: torch.Tensor
    hard_negative: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {key: float(getattr(self, key).detach()) for key in self.__dataclass_fields__}


class LossComputer:
    def __init__(self, config: dict) -> None:
        self.supervised_weight = float(config.get("supervised_weight", 1.0))
        self.contrastive_weight = float(config.get("supervised_contrastive_weight", 0.0))
        self.distillation_weight = float(config.get("distillation_weight", 0.0))
        self.false_negative_weight = float(config.get("false_negative_weight", 0.0))
        self.hard_negative_weight = float(config.get("hard_negative_weight", 0.0))
        self.positive_class_weight = float(config.get("positive_class_weight", 1.0))
        self.temperature = float(config.get("distillation_temperature", 3.0))
        self.label_smoothing = float(config.get("label_smoothing", 0.0))
        weights = (self.supervised_weight, self.contrastive_weight, self.distillation_weight, self.false_negative_weight, self.hard_negative_weight)
        if any(value < 0 for value in weights):
            raise ValueError("Loss weights must be non-negative")
        if self.positive_class_weight <= 0 or self.temperature <= 0:
            raise ValueError("positive_class_weight and distillation_temperature must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")

    def __call__(
        self,
        student: ModelOutput,
        labels: torch.Tensor,
        teacher: ModelOutput | None = None,
        hard_negative_mask: torch.Tensor | None = None,
    ) -> LossResult:
        zero = student.logits.sum() * 0
        supervised = F.cross_entropy(student.logits, labels, label_smoothing=self.label_smoothing)
        contrastive = supervised_contrastive_loss(student.features, labels) if self.contrastive_weight and student.features is not None else zero
        distillation = self._distillation(student.logits, teacher.logits) if self.distillation_weight and teacher is not None else zero
        unsafe_logit = torch.logsumexp(student.logits[:, 1:], dim=1) - student.logits[:, 0]
        unsafe_target = labels.ne(0).float()
        positive_weight = torch.tensor(self.positive_class_weight, device=labels.device)
        false_negative = F.binary_cross_entropy_with_logits(unsafe_logit, unsafe_target, pos_weight=positive_weight) if self.false_negative_weight else zero
        if self.hard_negative_weight and hard_negative_mask is not None and hard_negative_mask.any():
            hard_negative = F.binary_cross_entropy_with_logits(unsafe_logit[hard_negative_mask], torch.zeros_like(unsafe_logit[hard_negative_mask]))
        else:
            hard_negative = zero
        total = self.supervised_weight * supervised + self.contrastive_weight * contrastive + self.distillation_weight * distillation
        total = total + self.false_negative_weight * false_negative + self.hard_negative_weight * hard_negative
        return LossResult(total, supervised, contrastive, distillation, false_negative, hard_negative)

    def _distillation(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature * temperature
