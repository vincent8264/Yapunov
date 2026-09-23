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

    def show_spectrum(self, columns: tuple[int, ...]) -> None:
        """Render a low-priority 13-band audio spectrum when supported."""

    def show_notification_status(self, delivered: bool) -> None:
        """Show whether the current alert's notification was delivered, when supported."""

    def show_startup_status(self, status: str) -> None:
        """Render a board-specific startup state when supported."""

    def show_input_fault(self, reason: str, detail: str) -> None:
        """Show that the input is unavailable while recovery is attempted."""
        self.show_alert("microphone_fault")

    def clear_input_fault(self, reason: str) -> None:
        """Clear a previously displayed input fault after capture recovers."""
        self.clear_alert()

    def refresh_status(self) -> None:
        """Advance deferred status work without changing the current display."""

    @abstractmethod
    def apply_decision(self, decision: Decision) -> None: ...

    @abstractmethod
    def shutdown(self) -> None:
        """Best-effort transition of known switchable outputs to a safe state."""
