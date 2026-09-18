"""Future UNO Q Bridge backend.

The imports are intentionally local so this module remains importable on laptops.
Bridge calls follow current Arduino documentation but are untested until a board is
available; verify them against the App Lab version installed on the real UNO Q.
"""

from typing import Any

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend


class UnoQHardware(HardwareBackend):
    def __init__(self) -> None:
        try:
            from arduino.app_utils import Bridge
        except ImportError as exc:
            raise RuntimeError(
                "UnoQHardware requires the Arduino App Lab runtime on an UNO Q. "
                "Use MockHardware on Windows or macOS."
            ) from exc
        self._bridge: Any = Bridge

    def set_led(self, enabled: bool) -> None:
        self._bridge.call("set_led", enabled)

    def set_pwm(self, channel: int, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("PWM value must be between 0.0 and 1.0")
        self._bridge.call("set_pwm", channel, value)

    def move_servo(self, channel: int, degrees: float) -> None:
        if not 0.0 <= degrees <= 180.0:
            raise ValueError("servo degrees must be between 0 and 180")
        self._bridge.call("move_servo", channel, degrees)

    def apply_decision(self, decision: Decision) -> None:
        self.set_led(decision.action == "alert")
