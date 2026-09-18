"""Small, swappable input-to-hardware pipeline."""

from collections.abc import Callable
from typing import Any

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend
from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inputs.base import InputSource


class Pipeline:
    def __init__(
        self,
        input_source: InputSource,
        preprocessor: Callable[[Any], Any],
        inference: InferenceEngine,
        decision_function: Callable[[InferenceResult], Decision],
        hardware: HardwareBackend,
    ) -> None:
        self.input_source = input_source
        self.preprocessor = preprocessor
        self.inference = inference
        self.decision_function = decision_function
        self.hardware = hardware

    def step(self) -> tuple[InferenceResult, Decision]:
        data = self.input_source.read()
        processed = self.preprocessor(data)
        result = self.inference.predict(processed)
        decision = self.decision_function(result)
        self.hardware.apply_decision(decision)
        return result, decision
