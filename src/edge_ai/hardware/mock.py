"""Cross-platform hardware substitute with inspectable state."""

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.icons import render_icon


class MockHardware(HardwareBackend):
    def __init__(self, *, verbose: bool = True) -> None:
        self.verbose = verbose
        self.led_enabled = False
        self.pwm_values: dict[int, float] = {}
        self.servo_positions: dict[int, float] = {}
        self.last_decision: Decision | None = None
        self.current_alert: str | None = None
        self.alert_history: list[str] = []
        self.spectrum_columns: tuple[int, ...] = (0,) * 13
        self.spectrum_history: list[tuple[int, ...]] = []

    def _print(self, message: str) -> None:
        if self.verbose:
            print(f"[MOCK] {message}")

    def check_connection(self) -> None:
        self._print("CONNECTION OK")

    def set_led(self, enabled: bool) -> None:
        self.led_enabled = enabled
        self._print(f"LED {'ON' if enabled else 'OFF'}")

    def set_pwm(self, channel: int, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("PWM value must be between 0.0 and 1.0")
        self.pwm_values[channel] = value
        self._print(f"PWM {channel} -> {value:.2f}")

    def move_servo(self, channel: int, degrees: float) -> None:
        if not 0.0 <= degrees <= 180.0:
            raise ValueError("servo degrees must be between 0 and 180")
        self.servo_positions[channel] = degrees
        self._print(f"SERVO {channel} -> {degrees:.1f} degrees")

    def show_alert(self, event: str) -> None:
        self.current_alert = event
        self.alert_history.append(event)
        self._print(f"DISPLAY: {event}")
        try:
            preview = render_icon(event)
        except ValueError:
            preview = None
        if preview is not None:
            self._print(preview)
        self.set_led(True)

    def clear_alert(self) -> None:
        self.current_alert = None
        self.set_led(False)

    def show_spectrum(self, columns: tuple[int, ...]) -> None:
        if len(columns) != 13 or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 8
            for value in columns
        ):
            raise ValueError("audio spectrum must contain 13 integer levels from 0 through 8")
        if self.current_alert is not None:
            return
        self.spectrum_columns = columns
        self.spectrum_history.append(columns)
        self._print(f"SPECTRUM: {''.join(map(str, columns))}")

    def apply_decision(self, decision: Decision) -> None:
        self.last_decision = decision
        self._print(f"ACTION: {decision.action}")
        if decision.action == "alert":
            self.show_alert(decision.event or "alert")
        else:
            self.clear_alert()

    def shutdown(self) -> None:
        self.clear_alert()
        for channel in tuple(self.pwm_values):
            self.set_pwm(channel, 0.0)
