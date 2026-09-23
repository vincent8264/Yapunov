from typing import Any

import pytest

from edge_ai.hardware.mock import MockHardware
from edge_ai.inputs.base import InputSource
from edge_ai.startup import check_internet, run_startup_validation


class AudioInput(InputSource):
    def __init__(self) -> None:
        self.reads = 0

    def read(self) -> object:
        self.reads += 1
        return object()


def test_startup_validation_checks_board_audio_and_internet() -> None:
    hardware = MockHardware(verbose=False)
    input_source = AudioInput()
    stage: list[str] = []

    report = run_startup_validation(
        hardware,
        input_source,
        on_hardware_ready=lambda: stage.append("hardware"),
        internet_checker=lambda: (True, "HTTPS reachable (200)"),
    )

    assert input_source.reads == 1
    assert stage == ["hardware"]
    assert report.internet_available is True


def test_startup_validation_fails_when_microphone_cannot_capture() -> None:
    class SilentInput(InputSource):
        def read(self) -> object:
            raise RuntimeError("no samples")

    with pytest.raises(RuntimeError, match="microphone startup check failed: no samples"):
        run_startup_validation(
            MockHardware(verbose=False),
            SilentInput(),
            internet_checker=lambda: (True, "unused"),
        )


def test_internet_check_closes_response() -> None:
    class Response:
        status = 204
        closed = False

        def close(self) -> None:
            self.closed = True

    response = Response()

    def opener(request: object, *, timeout: float) -> Any:
        assert getattr(request, "full_url") == "https://example.test/"
        assert timeout == 1.0
        return response

    assert check_internet(opener=opener, url="https://example.test/", timeout_seconds=1.0) == (
        True,
        "HTTPS reachable (204)",
    )
    assert response.closed is True
