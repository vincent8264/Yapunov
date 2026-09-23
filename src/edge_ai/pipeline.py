"""Small, swappable input-to-hardware pipeline."""

from collections.abc import Callable
from typing import Any

from edge_ai.decision import Decision
from edge_ai.audio_display import AudioSpectrum
from edge_ai.hardware.base import HardwareBackend
from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inputs.base import InputSource
from edge_ai.inputs.audio import MicrophoneHealthError, MicrophoneHealthInput


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
        self._input_fault_active = False

    @property
    def poll_interval_seconds(self) -> float:
        return self.audio_spectrum.frame_duration_seconds if self.audio_spectrum else 0.0

    def step(self) -> tuple[InferenceResult, Decision] | None:
        try:
            data = self.input_source.read()
        except MicrophoneHealthError as exc:
            if not self._input_fault_active:
                self.hardware.show_input_fault(exc.reason, str(exc))
                self._input_fault_active = True
            else:
                self.hardware.refresh_status()
            return None
        if self._input_fault_active:
            recovered_reason = (
                self.input_source.take_recovered_reason()
                if isinstance(self.input_source, MicrophoneHealthInput)
                else None
            )
            if recovered_reason is None:
                self.hardware.refresh_status()
                return None
            self.hardware.clear_input_fault(recovered_reason)
            self._input_fault_active = False
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
