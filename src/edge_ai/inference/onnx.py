"""Thin ONNX Runtime adapter; model-specific output conversion stays here."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from edge_ai.inference.base import InferenceEngine, InferenceResult

OutputAdapter = Callable[[Sequence[Any]], InferenceResult]


class ONNXInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model_path: Path,
        *,
        labels: Sequence[str] | None = None,
        output_adapter: OutputAdapter | None = None,
        output_type: str | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        model_inputs = self._session.get_inputs()
        if len(model_inputs) != 1:
            raise ValueError(
                f"Starter adapter expects one model input, found {len(model_inputs)}; "
                "adapt ONNXInferenceEngine for this model"
            )
        self.input_name = model_inputs[0].name
        self._input_rank = len(model_inputs[0].shape)
        self._labels = tuple(labels) if labels is not None else None
        if output_adapter is None and output_type != "probabilities":
            raise ValueError(
                "The generic ONNX output adapter requires output_type='probabilities'. "
                "Confirm the model activation or provide a model-specific output_adapter."
            )
        self._output_adapter = output_adapter or self._default_output_adapter

    def predict(self, data: Any) -> InferenceResult:
        model_input = np.asarray(data, dtype=np.float32)
        if model_input.ndim + 1 == self._input_rank:
            model_input = np.expand_dims(model_input, axis=0)
        outputs = self._session.run(None, {self.input_name: model_input})
        return self._output_adapter(outputs)

    def _default_output_adapter(self, outputs: Sequence[Any]) -> InferenceResult:
        if not outputs:
            raise ValueError("ONNX model returned no outputs")
        scores = np.asarray(outputs[0], dtype=np.float32).squeeze().reshape(-1)
        if scores.size == 0:
            raise ValueError("ONNX model returned an empty first output")
        if scores.size == 1:
            score = float(scores[0])
            label = "anomaly" if score >= 0.5 else "normal"
            confidence = score if label == "anomaly" else 1.0 - score
            return InferenceResult(label, float(np.clip(confidence, 0.0, 1.0)))

        index = int(np.argmax(scores))
        if self._labels is not None and len(self._labels) != scores.size:
            raise ValueError(
                f"configured {len(self._labels)} labels for {scores.size} model scores"
            )
        label = self._labels[index] if self._labels and index < len(self._labels) else str(index)
        return InferenceResult(label, float(np.clip(scores[index], 0.0, 1.0)))
