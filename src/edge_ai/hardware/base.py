"""Hardware backend contract."""

from abc import ABC, abstractmethod

from edge_ai.decision import Decision


class HardwareBackend(ABC):
    @abstractmethod
    def check_connection(self) -> None:
        """Raise a useful error if the backend cannot reach its hardware."""

    @abstractmethod
    def set_led(self, enabled: bool) -> None: ...

    @abstractmethod
    def set_pwm(self, channel: int, value: float) -> None: ...

    @abstractmethod
    def move_servo(self, channel: int, degrees: float) -> None: ...

    def show_alert(self, event: str) -> None:
        """Display an event, falling back to the simplest available indicator."""
        self.set_led(True)

    def clear_alert(self) -> None:
        """Clear the current visual event."""
        self.set_led(False)

    @abstractmethod
    def apply_decision(self, decision: Decision) -> None: ...

    @abstractmethod
    def shutdown(self) -> None:
        """Best-effort transition of known switchable outputs to a safe state."""
