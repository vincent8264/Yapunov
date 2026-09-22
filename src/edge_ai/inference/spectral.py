"""Transparent spectral baseline used to validate plumbing before a model is selected."""

from typing import Any

from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.preprocessing.audio import AudioFeatures


class SpectralSoundInferenceEngine(InferenceEngine):
    """Classify distinct synthetic signatures; not a production safety model."""

    def __init__(self, *, min_rms: float = 0.02) -> None:
        if not 0.0 <= min_rms <= 1.0:
            raise ValueError("min_rms must be between 0 and 1")
        self.min_rms = min_rms

    def predict(self, data: Any) -> InferenceResult:
        if not isinstance(data, AudioFeatures):
            raise TypeError("spectral sound inference requires AudioFeatures")
        if data.rms < self.min_rms:
            confidence = min(1.0, 1.0 - data.rms / max(self.min_rms, 1e-9))
            return InferenceResult("background", confidence)
        if data.low_energy_ratio >= 0.55:
            return InferenceResult("fall_thud", min(0.99, 0.6 + data.low_energy_ratio * 0.4))
        if data.high_energy_ratio >= 0.50 and data.zero_crossing_rate >= 0.45:
            return InferenceResult("glass_break", min(0.99, 0.6 + data.high_energy_ratio * 0.4))
        if data.high_energy_ratio >= 0.55:
            return InferenceResult("smoke_alarm", min(0.99, 0.75 + data.high_energy_ratio * 0.25))
        return InferenceResult("background", 0.6)
