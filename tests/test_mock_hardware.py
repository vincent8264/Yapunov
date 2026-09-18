from edge_ai.decision import Decision
from edge_ai.hardware.mock import MockHardware


def test_apply_decision_updates_mock_state() -> None:
    hardware = MockHardware(verbose=False)

    hardware.apply_decision(Decision("alert"))
    assert hardware.last_decision == Decision("alert")
    assert hardware.led_enabled is True

    hardware.apply_decision(Decision("idle"))
    assert hardware.last_decision == Decision("idle")
    assert hardware.led_enabled is False
