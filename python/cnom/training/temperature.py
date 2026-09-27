from __future__ import annotations

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, initial: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(initial).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach())
