"""Replaceable, hardware-independent decision rules."""

from dataclasses import dataclass

from edge_ai.inference.base import InferenceResult


@dataclass(frozen=True)
class Decision:
    action: str
    value: float | None = None
    event: str | None = None
    confidence: float | None = None
    notify: bool = False


def decide(result: InferenceResult) -> Decision:
    if result.label == "anomaly" and result.confidence > 0.8:
        return Decision(action="alert")
    return Decision(action="idle")
