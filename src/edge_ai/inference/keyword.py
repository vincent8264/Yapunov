"""ONNX adapter for a local raw-waveform help keyword model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from edge_ai.inference.base import InferenceEngine, InferenceResult

ort.disable_telemetry_events()

SessionFactory = Callable[..., Any]


class KeywordSpotterInferenceEngine(InferenceEngine):
    """Classify a waveform with an ONNX ``help`` probability output."""

    def __init__(
        self,
        model_path: Path,
        *,
        activation_threshold: float = 0.5,
        positive_index: int = 1,
        session_factory: SessionFactory | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"keyword ONNX model not found: {path}")
        if not 0.0 <= activation_threshold <= 1.0:
            raise ValueError("keyword activation_threshold must be between 0 and 1")
        if isinstance(positive_index, bool) or positive_index < 0:
            raise ValueError("keyword positive_index must be non-negative")

        factory = session_factory or ort.InferenceSession
        self._session = factory(str(path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) not in {1, 2}:
            raise ValueError("keyword model must have one rank-1 or rank-2 waveform input")
        outputs = self._session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("keyword model must expose exactly one probability output")
        self.input_name = inputs[0].name
        self.input_rank = len(inputs[0].shape)
        self.output_name = outputs[0].name
        self.activation_threshold = activation_threshold
        self.positive_index = positive_index

    def predict(self, data: Any) -> InferenceResult:
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("keyword input must be a non-empty rank-1 waveform")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("keyword input waveform must contain only finite values")
        model_input = waveform if self.input_rank == 1 else waveform[np.newaxis, :]
        outputs = self._session.run([self.output_name], {self.input_name: model_input})
        probabilities = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if probabilities.size == 0:
            raise ValueError("keyword model output must contain at least one probability")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError(
                "keyword model output must contain probability values from 0 to 1"
            )
        if probabilities.size == 1:
            confidence = float(probabilities[0])
        elif self.positive_index < probabilities.size:
            confidence = float(probabilities[self.positive_index])
        else:
            raise ValueError(
                f"keyword positive_index {self.positive_index} is invalid for "
                f"{probabilities.size} output scores"
            )
        common = {
            "model_label": "help_call",
            "model_confidence": confidence,
            "source": "keyword_spotter",
            "evaluated_events": ("help_call",),
        }
        if confidence < self.activation_threshold:
            return InferenceResult("background", 1.0 - confidence, **common)
        return InferenceResult("help_call", confidence, **common)
