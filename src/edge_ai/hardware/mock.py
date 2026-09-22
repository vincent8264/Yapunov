"""Cross-platform hardware substitute with inspectable state."""

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend


class MockHardware(HardwareBackend):
    def __init__(self, *, verbose: bool = True) -> None:
        self.verbose = verbose
        self.led_enabled = False
        self.pwm_values: dict[int, float] = {}
        self.servo_positions: dict[int, float] = {}
        self.last_decision: Decision | None = None

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

    def apply_decision(self, decision: Decision) -> None:
        self.last_decision = decision
        self._print(f"ACTION: {decision.action}")
        self.set_led(decision.action == "alert")

    def shutdown(self) -> None:
        self.set_led(False)
        for channel in tuple(self.pwm_values):
            self.set_pwm(channel, 0.0)
