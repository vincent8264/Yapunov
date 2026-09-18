"""Model-free threshold inference for sensor demos and tests."""

from typing import Any

import numpy as np

from edge_ai.inference.base import InferenceEngine, InferenceResult


class DummyInferenceEngine(InferenceEngine):
    def __init__(self, threshold: float = 0.8) -> None:
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold

    def predict(self, data: Any) -> InferenceResult:
        values = np.asarray(data, dtype=np.float32)
        if values.size == 0:
            raise ValueError("data must contain at least one value")
        value = float(values.mean())
        label = "anomaly" if value >= self.threshold else "normal"
        distance = abs(value - self.threshold) / self.threshold
        confidence = min(1.0, 0.8 + 0.2 * distance)
        return InferenceResult(label=label, confidence=confidence)
