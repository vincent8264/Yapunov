"""Rolling-history warnings for slowly developing risks such as leaks or cooking."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
import time

from edge_ai.decision import Decision, RiskWarning
from edge_ai.inference.base import InferenceResult


@dataclass(frozen=True)
class RiskRule:
    """A warning raised when any of ``labels`` persists across a rolling window."""

    name: str
    message: str
    labels: tuple[str, ...]
    threshold: float
    window_seconds: float = 15.0
    min_detected_seconds: float = 7.0
    cooldown_seconds: float = 300.0

    def validated(self) -> "RiskRule":
        if not self.name:
            raise ValueError("risk name must not be empty")
        if not self.message:
            raise ValueError(f"risk {self.name!r} message must not be empty")
        if not self.labels or any(not label for label in self.labels):
            raise ValueError(f"risk {self.name!r} must list at least one model label")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"risk {self.name!r} threshold must be between 0 and 1")
        if self.window_seconds <= 0.0:
            raise ValueError(f"risk {self.name!r} window_seconds must be positive")
        if not 0.0 < self.min_detected_seconds <= self.window_seconds:
            raise ValueError(
                f"risk {self.name!r} min_detected_seconds must be positive and no "
                "longer than window_seconds"
            )
        if self.cooldown_seconds < 0.0:
            raise ValueError(f"risk {self.name!r} cooldown_seconds must be non-negative")
        return self


@dataclass(frozen=True)
class RiskObservation:
    timestamp: float
    duration_seconds: float
    detected: bool
    label: str | None
    score: float


class RollingRiskMonitor:
    """Track timestamped label detections and warn when one persists long enough.

    Each observation is credited with the time since the previous one, capped at
    ``max_observation_seconds`` so a stalled input cannot count as sustained sound.
    The first observation is credited with no time.
    """

    def __init__(
        self,
        rules: Sequence[RiskRule],
        *,
        max_observation_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not rules:
            raise ValueError("risk monitor requires at least one rule")
        validated = [rule.validated() for rule in rules]
        names = [rule.name for rule in validated]
        if len(set(names)) != len(names):
            raise ValueError("risk names must be unique")
        if max_observation_seconds <= 0.0:
            raise ValueError("max_observation_seconds must be positive")
        self.rules = tuple(validated)
        self.max_observation_seconds = max_observation_seconds
        self._clock = clock
        self._history: dict[str, deque[RiskObservation]] = {
            rule.name: deque() for rule in self.rules
        }
        self._last_observation: float | None = None
        self._last_warning: dict[str, float] = {}

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(label for rule in self.rules for label in rule.labels))

    def history(self, name: str) -> tuple[RiskObservation, ...]:
        return tuple(self._history[name])

    def detected_seconds(self, name: str) -> float:
        return sum(
            item.duration_seconds for item in self._history[name] if item.detected
        )

    def update(self, label_scores: Mapping[str, float]) -> tuple[RiskRule, ...]:
        """Record one set of label scores and return warnings that start now."""
        now = self._clock()
        duration = (
            0.0
            if self._last_observation is None
            else min(max(now - self._last_observation, 0.0), self.max_observation_seconds)
        )
        self._last_observation = now

        triggered: list[RiskWarning] = []
        for rule in self.rules:
            label, score = max(
                ((label, label_scores.get(label, 0.0)) for label in rule.labels),
                key=lambda item: item[1],
            )
            detected = score >= rule.threshold
            history = self._history[rule.name]
            history.append(
                RiskObservation(now, duration, detected, label if detected else None, score)
            )
            while history and history[0].timestamp <= now - rule.window_seconds:
                history.popleft()

            detected_seconds = self.detected_seconds(rule.name)
            if detected_seconds < rule.min_detected_seconds:
                continue
            last = self._last_warning.get(rule.name, -math.inf)
            if now - last < rule.cooldown_seconds:
                continue
            self._last_warning[rule.name] = now
            peak = max(
                (item for item in history if item.detected), key=lambda item: item.score
            )
            triggered.append(
                RiskWarning(
                    risk=rule.name,
                    message=rule.message,
                    detected_seconds=detected_seconds,
                    window_seconds=rule.window_seconds,
                    peak_score=peak.score,
                    peak_label=peak.label,
                )
            )
        return tuple(triggered)


class RiskWarningDecision:
    """Add rolling risk warnings to an unchanged emergency decision function."""

    def __init__(
        self,
        emergency_decision: Callable[[InferenceResult], Decision],
        monitor: RollingRiskMonitor,
    ) -> None:
        self.emergency_decision = emergency_decision
        self.monitor = monitor

    def __call__(self, result: InferenceResult) -> Decision:
        decision = self.emergency_decision(result)
        if not result.label_scores:
            return decision
        warnings = self.monitor.update(dict(result.label_scores))
        if not warnings:
            return decision
        return replace(decision, warnings=warnings)
