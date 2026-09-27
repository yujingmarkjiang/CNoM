from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class TemporalPolicy:
    frame_threshold: float = 0.30
    window_frames: int = 20
    persistence: float = 0.55
    merge_seconds: float = 3.0
    cooldown_seconds: float = 8.0
    pre_roll_s: float = 1.6
    post_roll_s: float = 1.6
    max_clip_s: float = 12.0

    def __post_init__(self) -> None:
        if not 0 <= self.frame_threshold <= 1 or not 0 < self.persistence <= 1:
            raise ValueError("Temporal thresholds must be within [0, 1]")
        if self.window_frames <= 0 or min(self.merge_seconds, self.cooldown_seconds, self.pre_roll_s, self.post_roll_s) < 0 or self.max_clip_s <= 0:
            raise ValueError("Temporal durations and window sizes must be valid")


@dataclass(frozen=True)
class StreamingTemporalPolicy:
    frame_threshold: float = 0.30
    window_frames: int = 20
    persistence: float = 0.55
    pre_roll_s: float = 2.0
    decision_post_roll_s: float = 0.4
    max_clip_s: float = 5.0
    reset_persistence: float = 0.20
    reset_seconds: float = 2.0
    cooldown_seconds: float = 8.0

    def __post_init__(self) -> None:
        if not 0 <= self.frame_threshold <= 1 or not 0 < self.persistence <= 1 or not 0 <= self.reset_persistence < self.persistence:
            raise ValueError("Streaming thresholds must satisfy 0 <= reset < persistence <= 1")
        if self.window_frames <= 0 or min(self.pre_roll_s, self.decision_post_roll_s, self.reset_seconds, self.cooldown_seconds) < 0 or self.max_clip_s <= 0:
            raise ValueError("Streaming durations and window sizes must be valid")


@dataclass(frozen=True)
class CandidateEvent:
    candidate_id: str
    camera_id: str
    source_id: str
    start_s: float
    end_s: float
    trigger_s: float
    max_score: float
    mean_score: float
    frames_above: int
    frames_total: int

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.camera_id or not self.source_id:
            raise ValueError("Candidate identifiers must be non-empty")
        values = (self.start_s, self.end_s, self.trigger_s, self.max_score, self.mean_score)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Candidate times and scores must be finite")
        if self.start_s < 0 or self.end_s < self.start_s or not self.start_s <= self.trigger_s <= self.end_s:
            raise ValueError("Candidate timing is invalid")
        if not 0 <= self.mean_score <= 1 or not 0 <= self.max_score <= 1 or self.mean_score > self.max_score:
            raise ValueError("Candidate scores must satisfy 0 <= mean <= max <= 1")
        if self.frames_total <= 0 or not 0 <= self.frames_above <= self.frames_total:
            raise ValueError("Candidate frame counts are invalid")


class TemporalGate:
    def __init__(self, camera_id: str, source_id: str, policy: TemporalPolicy) -> None:
        self.camera_id, self.source_id, self.policy = camera_id, source_id, policy
        self.window: deque[tuple[float, float]] = deque(maxlen=policy.window_frames)
        self.active_start: float | None = None
        self.active_trigger: float | None = None
        self.active_last: float | None = None
        self.active_scores: list[float] = []
        self.cooldown_until = float("-inf")
        self.last_end_s = float("-inf")
        self.last_start_s = 0.0
        self.last_trigger_s = 0.0
        self.sequence = 0
        self.last_timestamp_s = float("-inf")

    def update(self, timestamp_s: float, score: float) -> list[CandidateEvent]:
        _validate_observation(timestamp_s, score, self.last_timestamp_s)
        self.last_timestamp_s = timestamp_s
        self.window.append((timestamp_s, score))
        if len(self.window) < self.policy.window_frames:
            return []
        persistence = sum(value > self.policy.frame_threshold for _, value in self.window) / len(self.window)
        if persistence >= self.policy.persistence and timestamp_s >= self.cooldown_until:
            if self.active_start is None:
                self.active_start = self.window[0][0]
                self.active_trigger = timestamp_s
                self.active_scores = [value for _, value in self.window]
            else:
                self.active_scores.append(score)
            self.active_last = timestamp_s
            return []
        if self.active_start is not None and persistence < self.policy.persistence:
            if timestamp_s - self.active_last <= self.policy.merge_seconds:
                return []
            return [self._close(self.active_last)]
        return []

    def flush(self, final_timestamp_s: float | None = None) -> list[CandidateEvent]:
        if self.active_start is None:
            return []
        return [self._close(final_timestamp_s if final_timestamp_s is not None else self.active_last)]

    def _close(self, close_s: float | None = None) -> CandidateEvent:
        if self.active_start is None or self.active_last is None or self.active_trigger is None:
            raise RuntimeError("No active event")
        close_s = self.active_last if close_s is None else close_s
        end = min(self.active_start + self.policy.max_clip_s, max(self.active_trigger, close_s + self.policy.post_roll_s))
        self.sequence += 1
        event = CandidateEvent(
            candidate_id=f"{self.source_id}:{self.sequence:06d}", camera_id=self.camera_id, source_id=self.source_id,
            start_s=max(0.0, self.active_start - self.policy.pre_roll_s), end_s=end, trigger_s=self.active_trigger,
            max_score=max(self.active_scores),
            mean_score=sum(self.active_scores) / len(self.active_scores),
            frames_above=sum(value > self.policy.frame_threshold for value in self.active_scores),
            frames_total=len(self.active_scores),
        )
        self.last_end_s = close_s
        self.last_start_s = self.active_start
        self.last_trigger_s = self.active_trigger
        self.cooldown_until = close_s + self.policy.cooldown_seconds
        self.active_start = self.active_trigger = self.active_last = None
        self.active_scores = []
        return event


class StreamingTemporalGate:
    def __init__(self, camera_id: str, source_id: str, policy: StreamingTemporalPolicy) -> None:
        self.camera_id, self.source_id, self.policy = camera_id, source_id, policy
        self.window: deque[tuple[float, float]] = deque(maxlen=policy.window_frames)
        self.pending: dict[str, float | list[float]] | None = None
        self.latched = False
        self.clear_since: float | None = None
        self.cooldown_until = float("-inf")
        self.sequence = 0
        self.last_timestamp_s = float("-inf")

    def update(self, timestamp_s: float, score: float) -> list[CandidateEvent]:
        _validate_observation(timestamp_s, score, self.last_timestamp_s)
        self.last_timestamp_s = timestamp_s
        self.window.append((timestamp_s, score))
        completed: list[CandidateEvent] = []
        if len(self.window) < self.policy.window_frames:
            return completed
        values = [value for _, value in self.window]
        persistence = sum(value > self.policy.frame_threshold for value in values) / len(values)
        self._update_latch(timestamp_s, persistence)
        if self.pending is None and not self.latched and timestamp_s >= self.cooldown_until and persistence >= self.policy.persistence:
            trigger_s = timestamp_s
            start_s = max(0.0, self.window[0][0] - self.policy.pre_roll_s)
            end_s = max(trigger_s, min(start_s + self.policy.max_clip_s, trigger_s + self.policy.decision_post_roll_s))
            self.pending = {"start_s": start_s, "end_s": end_s, "trigger_s": trigger_s, "scores": list(values)}
            self.latched = True
        elif self.pending is not None:
            scores = self.pending["scores"]
            if isinstance(scores, list):
                scores.append(score)
        if self.pending is not None and timestamp_s >= float(self.pending["end_s"]):
            completed.append(self._emit())
        return completed

    def flush(self, final_timestamp_s: float | None = None) -> list[CandidateEvent]:
        if self.pending is None:
            return []
        if final_timestamp_s is not None:
            self.pending["end_s"] = min(float(self.pending["end_s"]), final_timestamp_s)
        return [self._emit()]

    def _update_latch(self, timestamp_s: float, persistence: float) -> None:
        if not self.latched:
            self.clear_since = None
            return
        if persistence <= self.policy.reset_persistence:
            self.clear_since = timestamp_s if self.clear_since is None else self.clear_since
            if timestamp_s - self.clear_since >= self.policy.reset_seconds:
                self.latched = False
                self.clear_since = None
        else:
            self.clear_since = None

    def _emit(self) -> CandidateEvent:
        if self.pending is None:
            raise RuntimeError("No pending decision clip")
        scores = self.pending["scores"]
        if not isinstance(scores, list) or not scores:
            raise RuntimeError("Pending decision clip has no scores")
        self.sequence += 1
        event = CandidateEvent(
            candidate_id=f"{self.source_id}:{self.sequence:06d}", camera_id=self.camera_id, source_id=self.source_id,
            start_s=float(self.pending["start_s"]), end_s=float(self.pending["end_s"]), trigger_s=float(self.pending["trigger_s"]),
            max_score=max(scores), mean_score=sum(scores) / len(scores), frames_above=sum(value > self.policy.frame_threshold for value in scores),
            frames_total=len(scores),
        )
        self.cooldown_until = event.trigger_s + self.policy.cooldown_seconds
        self.pending = None
        return event


def generate_candidates(frame_scores: pd.DataFrame, policy: TemporalPolicy) -> pd.DataFrame:
    required = {"camera_id", "source_id", "timestamp_s", "score"}
    missing = required.difference(frame_scores.columns)
    if missing:
        raise ValueError(f"Frame-score table lacks columns: {sorted(missing)}")
    events: list[dict] = []
    for (camera_id, source_id), group in frame_scores.sort_values("timestamp_s").groupby(["camera_id", "source_id"], sort=False):
        gate = TemporalGate(str(camera_id), str(source_id), policy)
        for row in group.itertuples():
            events.extend(event.__dict__ for event in gate.update(float(row.timestamp_s), float(row.score)))
        events.extend(event.__dict__ for event in gate.flush(float(group["timestamp_s"].max())))
    return pd.DataFrame(events, columns=CandidateEvent.__dataclass_fields__.keys())


def _validate_observation(timestamp_s: float, score: float, previous_timestamp_s: float) -> None:
    if not math.isfinite(timestamp_s) or timestamp_s < 0 or timestamp_s < previous_timestamp_s:
        raise ValueError("Frame timestamps must be finite, non-negative, and non-decreasing")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Frame scores must be finite and within [0, 1]")
