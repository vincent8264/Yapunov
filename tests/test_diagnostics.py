from pathlib import Path

from edge_ai.config import ActuatorCheck, HardwareCheckPlan, load_hardware_check_config
from edge_ai.diagnostics import run_hardware_checks
from edge_ai.hardware.mock import MockHardware


def test_mock_hardware_check_passes_and_cleans_up() -> None:
    plan = load_hardware_check_config(Path("configs/hardware-check.toml"))

    result = run_hardware_checks(plan, sleep=lambda _: None)

    assert result.passed is True
    assert [item.status for item in result.checks] == ["passed", "passed"]
    assert plan.hardware.led_enabled is False  # type: ignore[attr-defined]


def test_actuators_are_opt_in() -> None:
    hardware = MockHardware(verbose=False)
    plan = HardwareCheckPlan(
        hardware=hardware,
        check_led=False,
        led_hold_seconds=0.0,
        actuators=(ActuatorCheck("pwm", 2, 0.25, 0.0),),
    )

    skipped = run_hardware_checks(plan)
    assert skipped.checks[1].status == "skipped"
    assert hardware.pwm_values == {}

    executed = run_hardware_checks(plan, allow_actuators=True)
    assert executed.checks[1].status == "passed"
    assert hardware.pwm_values[2] == 0.0


def test_connection_failure_is_reported() -> None:
    class DisconnectedHardware(MockHardware):
        def check_connection(self) -> None:
            raise RuntimeError("not reachable")

    plan = HardwareCheckPlan(DisconnectedHardware(verbose=False), False, 0.0, ())

    result = run_hardware_checks(plan)

    assert result.passed is False
    assert result.checks[0].status == "failed"
    assert "not reachable" in result.checks[0].detail
