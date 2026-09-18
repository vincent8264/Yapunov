from edge_ai.decision import Decision, decide
from edge_ai.hardware.mock import MockHardware
from edge_ai.inference.base import InferenceEngine, InferenceResult
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
