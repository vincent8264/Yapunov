import numpy as np

from edge_ai.audio_display import AudioSpectrum
from edge_ai.decision import Decision, decide
from edge_ai.hardware.mock import MockHardware
from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inputs.audio import AudioFrame
from edge_ai.inputs.base import InputSource
from edge_ai.pipeline import Pipeline


class FixedInput(InputSource):
    def read(self) -> float:
        return 2.0


class RecordingInference(InferenceEngine):
    def __init__(self) -> None:
        self.received: float | None = None

    def predict(self, data: float) -> InferenceResult:
        self.received = data
        return InferenceResult("anomaly", 0.95)


def test_pipeline_runs_all_stages() -> None:
    inference = RecordingInference()
    hardware = MockHardware(verbose=False)
    pipeline = Pipeline(FixedInput(), lambda value: value / 2, inference, decide, hardware)

    result, decision = pipeline.step()

    assert inference.received == 1.0
    assert result == InferenceResult("anomaly", 0.95)
    assert decision == Decision("alert")
    assert hardware.last_decision == decision
    assert hardware.led_enabled is True


def test_audio_display_ticks_before_one_second_inference_window() -> None:
    class ShortAudioInput(InputSource):
        def read(self) -> AudioFrame:
            return AudioFrame(np.full(800, 0.1, dtype=np.float32), 16_000)

    hardware = MockHardware(verbose=False)
    pipeline = Pipeline(
        ShortAudioInput(), lambda value: value, RecordingInference(), decide, hardware,
        audio_spectrum=AudioSpectrum(rate_hz=20, inference_duration_seconds=1.0),
    )

    for _ in range(19):
        assert pipeline.step() is None
    assert pipeline.step() is not None
    assert len(hardware.spectrum_history) == 20
