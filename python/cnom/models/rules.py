from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuleResult:
    unsafe: bool
    label: str
    score: float
    evidence: list[str]


class RuleBaseline:
    def __init__(self, class_rules: dict[str, list[str]], persistence_frames: int = 20, persistence_ratio: float = 0.55) -> None:
        if persistence_frames <= 0 or not 0 < persistence_ratio <= 1:
            raise ValueError("Rule persistence must use a positive window and a ratio in (0, 1]")
        if not class_rules or any(not str(label).strip() or not signals for label, signals in class_rules.items()):
            raise ValueError("Each rule class must define at least one required signal")
        self.class_rules = {str(label): [str(signal) for signal in signals] for label, signals in class_rules.items()}
        self.persistence_frames = persistence_frames
        self.persistence_ratio = persistence_ratio
        self.history: list[dict[str, bool]] = []

    def update(self, signals: dict[str, bool | float]) -> RuleResult:
        normalized = {key: bool(value) for key, value in signals.items()}
        self.history.append(normalized)
        self.history = self.history[-self.persistence_frames:]
        best = RuleResult(False, "safe_background", 0.0, [])
        for label, required in self.class_rules.items():
            votes = [all(frame.get(signal, False) for signal in required) for frame in self.history]
            score = sum(votes) / max(1, len(votes))
            if score >= self.persistence_ratio and score > best.score:
                best = RuleResult(True, label, score, required)
        return best
