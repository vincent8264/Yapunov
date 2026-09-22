from edge_ai.config import ConfiguredPipeline
from edge_ai.decision import decide
from edge_ai.hardware.mock import MockHardware
from edge_ai.inference.dummy import DummyInferenceEngine
from edge_ai.inputs.base import InputSource
from edge_ai.pipeline import Pipeline
from edge_ai.runner import run_pipeline


class CloseableInput(InputSource):
    def __init__(self) -> None:
        self.closed = False

    def read(self) -> float:
        return 1.0

    def close(self) -> None:
        self.closed = True


def test_runner_stops_at_limit_and_cleans_up() -> None:
    input_source = CloseableInput()
    hardware = MockHardware(verbose=False)
    pipeline = Pipeline(
        input_source,
        lambda value: value,
        DummyInferenceEngine(),
        decide,
        hardware,
    )
    configured = ConfiguredPipeline(pipeline, interval_seconds=0.0)
    output: list[str] = []

    steps = run_pipeline(configured, max_steps=2, emit=output.append)

    assert steps == 2
    assert len(output) == 2
    assert "label=anomaly" in output[0]
    assert input_source.closed is True
    assert hardware.led_enabled is False


def test_runner_checks_connection_before_reading_input() -> None:
    class DisconnectedHardware(MockHardware):
        def check_connection(self) -> None:
            raise RuntimeError("offline")

    input_source = CloseableInput()
    hardware = DisconnectedHardware(verbose=False)
    pipeline = Pipeline(input_source, lambda value: value, DummyInferenceEngine(), decide, hardware)

    try:
        run_pipeline(ConfiguredPipeline(pipeline, 0.0), max_steps=1)
    except RuntimeError as exc:
        assert str(exc) == "offline"
    else:
        raise AssertionError("connection failure was not raised")

    assert input_source.closed is True
