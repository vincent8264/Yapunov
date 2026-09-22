"""Safe, explicit hardware diagnostics."""

from collections.abc import Callable
from dataclasses import dataclass
import time

from edge_ai.config import ActuatorCheck, HardwareCheckPlan


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class HardwareCheckResult:
    passed: bool
    checks: tuple[CheckResult, ...]


def _attempt(name: str, operation: Callable[[], None]) -> CheckResult:
    try:
        operation()
    except Exception as exc:
        return CheckResult(name, "failed", f"{type(exc).__name__}: {exc}")
    return CheckResult(name, "passed", "ok")


def _run_actuator(plan: HardwareCheckPlan, actuator: ActuatorCheck) -> None:
    if actuator.kind == "pwm":
        try:
            plan.hardware.set_pwm(actuator.channel, actuator.value)
        finally:
            plan.hardware.set_pwm(actuator.channel, actuator.safe_value)
    else:
        try:
            plan.hardware.move_servo(actuator.channel, actuator.value)
        finally:
            plan.hardware.move_servo(actuator.channel, actuator.safe_value)


def run_hardware_checks(
    plan: HardwareCheckPlan,
    *,
    allow_actuators: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> HardwareCheckResult:
    """Run configured checks and always attempt to leave switchable outputs off."""
    results = [_attempt("connection", plan.hardware.check_connection)]

    if plan.check_led:
        def check_led() -> None:
            try:
                plan.hardware.set_led(True)
                sleep(plan.led_hold_seconds)
            finally:
                plan.hardware.set_led(False)

        results.append(_attempt("led", check_led))

    for actuator in plan.actuators:
        name = f"{actuator.kind}[{actuator.channel}]"
        if allow_actuators:
            results.append(_attempt(name, lambda item=actuator: _run_actuator(plan, item)))
        else:
            results.append(CheckResult(name, "skipped", "requires --allow-actuators"))

    cleanup = _attempt("cleanup", plan.hardware.shutdown)
    if cleanup.status == "failed":
        results.append(cleanup)

    return HardwareCheckResult(
        passed=all(result.status != "failed" for result in results),
        checks=tuple(results),
    )
