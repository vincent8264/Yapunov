"""Future UNO Q Bridge backend.

The imports are intentionally local so this module remains importable on laptops.
Bridge calls follow current Arduino documentation but are untested until a board is
available; verify them against the App Lab version installed on the real UNO Q.
"""

from typing import Any

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.icons import ICONS


class UnoQHardware(HardwareBackend):
    def __init__(self, bridge: Any | None = None) -> None:
        self._alert_active = False
        if bridge is not None:
            self._bridge = bridge
            return
        try:
            from arduino.app_utils import Bridge
        except ImportError as exc:
            raise RuntimeError(
                "UnoQHardware requires the Arduino App Lab runtime on an UNO Q. "
                "Use MockHardware on Windows or macOS."
            ) from exc
        self._bridge = Bridge

    def check_connection(self) -> None:
        response = self._bridge.call("health_check")
        if response != "ok":
            raise RuntimeError(f"UNO Q Bridge health check returned {response!r}")

    def set_led(self, enabled: bool) -> None:
        self._bridge.call("set_led", enabled)

    def show_alert(self, event: str) -> None:
        if event not in ICONS:
            raise ValueError(f"unsupported visual alert: {event!r}")
        response = self._bridge.call("show_alert", event)
        if response != "ok":
            raise RuntimeError(f"UNO Q display alert returned {response!r}")
        self._alert_active = True

    def clear_alert(self) -> None:
        response: Any = None
        try:
            response = self._bridge.call("clear_alert")
        finally:
            self.set_led(False)
        if response != "ok":
            raise RuntimeError(f"UNO Q clear alert returned {response!r}")
        self._alert_active = False

    def show_notification_status(self, delivered: bool) -> None:
        if not self._alert_active:
            return
        response = self._bridge.call("set_email_status", delivered)
        if response != "ok":
            raise RuntimeError(f"UNO Q email status returned {response!r}")

    def show_spectrum(self, columns: tuple[int, ...]) -> None:
        if len(columns) != 13 or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 8
            for value in columns
        ):
            raise ValueError("audio spectrum must contain 13 integer levels from 0 through 8")
        if self._alert_active:
            return
        response = self._bridge.call("show_spectrum", "".join(str(value) for value in columns))
        if response != "ok":
            raise RuntimeError(f"UNO Q display spectrum returned {response!r}")

    def set_pwm(self, channel: int, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("PWM value must be between 0.0 and 1.0")
        raise RuntimeError(
            "PWM is not enabled in the pre-event UNO Q adapter. Verify the board pin, "
            "voltage, wiring, and Bridge endpoint before implementing it."
        )

    def move_servo(self, channel: int, degrees: float) -> None:
        if not 0.0 <= degrees <= 180.0:
            raise ValueError("servo degrees must be between 0 and 180")
        raise RuntimeError(
            "Servo control is not enabled in the pre-event UNO Q adapter. Verify the "
            "board pin, power, travel limits, library, and Bridge endpoint first."
        )

    def apply_decision(self, decision: Decision) -> None:
        if decision.action == "alert":
            if decision.event is None:
                self.set_led(True)
                self._alert_active = True
            else:
                self.show_alert(decision.event)
        else:
            self.clear_alert()

    def shutdown(self) -> None:
        self.clear_alert()
