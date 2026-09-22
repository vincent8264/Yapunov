"""Inference contract and model-independent output."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    label: str
    confidence: float
    model_label: str | None = None
    model_confidence: float | None = None
    source: str | None = None
    evaluated_events: tuple[str, ...] | None = None
    timings_ms: tuple[tuple[str, float], ...] = ()


class InferenceEngine(ABC):
    @abstractmethod
    def predict(self, data: Any) -> InferenceResult:
        """Run inference and return a model-independent result."""
