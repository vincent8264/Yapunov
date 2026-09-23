"""Replaceable, hardware-independent decision rules."""

from dataclasses import dataclass

from edge_ai.inference.base import InferenceResult


@dataclass(frozen=True)
class RiskWarning:
    risk: str
    message: str
    detected_seconds: float
    window_seconds: float
    peak_score: float
    peak_label: str | None = None


@dataclass(frozen=True)
class Decision:
    action: str
    value: float | None = None
    event: str | None = None
    confidence: float | None = None
    notify: bool = False
    warnings: tuple[RiskWarning, ...] = ()


def decide(result: InferenceResult) -> Decision:
    if result.label == "anomaly" and result.confidence > 0.8:
        return Decision(action="alert")
    return Decision(action="idle")
