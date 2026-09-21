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

    @abstractmethod
    def apply_decision(self, decision: Decision) -> None: ...
