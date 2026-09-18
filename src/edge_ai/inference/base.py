"""Inference contract and model-independent output."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    label: str
    confidence: float


class InferenceEngine(ABC):
    @abstractmethod
    def predict(self, data: Any) -> InferenceResult:
        """Run inference and return a model-independent result."""
