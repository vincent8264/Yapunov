"""Small, swappable input-to-hardware pipeline."""

from collections.abc import Callable
from typing import Any

from edge_ai.decision import Decision
from edge_ai.audio_display import AudioSpectrum
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
        audio_spectrum: AudioSpectrum | None = None,
    ) -> None:
        self.input_source = input_source
        self.preprocessor = preprocessor
        self.inference = inference
        self.decision_function = decision_function
        self.hardware = hardware
        self.audio_spectrum = audio_spectrum

    @property
    def poll_interval_seconds(self) -> float:
        return self.audio_spectrum.frame_duration_seconds if self.audio_spectrum else 0.0

    def step(self) -> tuple[InferenceResult, Decision] | None:
        data = self.input_source.read()
        if self.audio_spectrum is not None:
            if not hasattr(data, "samples") or not hasattr(data, "sample_rate"):
                raise TypeError("audio spectrum requires an AudioFrame input")
            self.hardware.show_spectrum(self.audio_spectrum.update(data))
            data = self.audio_spectrum.append_for_inference(data)
            if data is None:
                return None
        processed = self.preprocessor(data)
        result = self.inference.predict(processed)
        decision = self.decision_function(result)
        self.hardware.apply_decision(decision)
        return result, decision
