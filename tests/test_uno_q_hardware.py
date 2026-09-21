from typing import Any

import pytest

from edge_ai.hardware.uno_q import UnoQHardware


class FakeBridge:
    def __init__(self, response: Any = "ok") -> None:
        self.response = response
        self.calls: list[tuple[Any, ...]] = []

    def call(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.response


def test_health_check_calls_bridge_endpoint() -> None:
    bridge = FakeBridge()
    hardware = UnoQHardware(bridge=bridge)

    hardware.check_connection()

    assert bridge.calls == [("health_check",)]


def test_health_check_rejects_unexpected_response() -> None:
    hardware = UnoQHardware(bridge=FakeBridge("wrong board"))

    with pytest.raises(RuntimeError, match="wrong board"):
        hardware.check_connection()
