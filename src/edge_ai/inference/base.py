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
    transcript: str | None = None


class InferenceEngine(ABC):
    # Streaming engines consume every microphone frame exactly once and maintain
    # any longer inference windows internally.  The pipeline uses this to keep
    # a display's rolling window from being replayed into an ASR stream.
    accepts_streaming_audio_frames = False

    @abstractmethod
    def predict(self, data: Any) -> InferenceResult:
        """Run inference and return a model-independent result."""
