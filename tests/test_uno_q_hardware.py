from typing import Any

import pytest

from edge_ai.decision import Decision
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


def test_unverified_actuators_are_guarded_without_bridge_calls() -> None:
    bridge = FakeBridge()
    hardware = UnoQHardware(bridge=bridge)

    with pytest.raises(RuntimeError, match="not enabled"):
        hardware.set_pwm(0, 0.5)
    with pytest.raises(RuntimeError, match="not enabled"):
        hardware.move_servo(0, 90.0)

    assert bridge.calls == []


def test_sound_decision_uses_matrix_bridge_endpoint() -> None:
    bridge = FakeBridge()
    hardware = UnoQHardware(bridge=bridge)

    hardware.apply_decision(Decision("alert", event="glass_break", confidence=0.9))
    hardware.apply_decision(Decision("idle"))

    assert bridge.calls == [
        ("show_alert", "glass_break"),
        ("clear_alert",),
        ("set_led", False),
    ]


def test_unknown_visual_alert_is_rejected_before_bridge_call() -> None:
    bridge = FakeBridge()
    hardware = UnoQHardware(bridge=bridge)

    with pytest.raises(ValueError, match="unsupported visual alert"):
        hardware.show_alert("unknown")

    assert bridge.calls == []


def test_shutdown_clears_matrix_and_builtin_led() -> None:
    bridge = FakeBridge()
    hardware = UnoQHardware(bridge=bridge)

    hardware.shutdown()

    assert bridge.calls == [("clear_alert",), ("set_led", False)]


def test_generic_alert_falls_back_to_builtin_led() -> None:
    bridge = FakeBridge()
    hardware = UnoQHardware(bridge=bridge)

    hardware.apply_decision(Decision("alert"))

    assert bridge.calls == [("set_led", True)]
