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


def test_sound_alert_records_visual_event() -> None:
    hardware = MockHardware(verbose=False)

    hardware.apply_decision(Decision("alert", event="fall_thud", confidence=0.9))

    assert hardware.current_alert == "fall_thud"
    assert hardware.alert_history == ["fall_thud"]
    hardware.shutdown()
    assert hardware.current_alert is None


def test_shutdown_turns_off_switchable_outputs() -> None:
    hardware = MockHardware(verbose=False)
    hardware.set_led(True)
    hardware.set_pwm(2, 0.7)

    hardware.shutdown()

    assert hardware.led_enabled is False
    assert hardware.pwm_values[2] == 0.0
