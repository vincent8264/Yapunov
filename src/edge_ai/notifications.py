"""Metadata-only remote notifications for locally classified events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
import queue
import smtplib
import threading
from typing import Final

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend


@dataclass(frozen=True)
class AlertNotification:
    event: str
    confidence: float
    device_name: str
    timestamp: str


class Notifier(ABC):
    @abstractmethod
    def notify(self, alert: AlertNotification) -> None: ...

    def close(self) -> None:
        pass


class SMTPNotifier(Notifier):
    """Send event metadata through SMTP; audio is not accepted by this API."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        sender: str,
        recipient: str,
        username: str | None = None,
        password: str | None = None,
        starttls: bool = True,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.sender = sender
        self.recipient = recipient
        self.username = username
        self.password = password
        self.starttls = starttls
        self.timeout_seconds = timeout_seconds

    def notify(self, alert: AlertNotification) -> None:
        message = EmailMessage()
        message["Subject"] = f"Safety sound detected: {alert.event.replace('_', ' ')}"
        message["From"] = self.sender
        message["To"] = self.recipient
        message.set_content(
            f"Device: {alert.device_name}\n"
            f"Event: {alert.event}\n"
            f"Confidence: {alert.confidence:.1%}\n"
            f"Time: {alert.timestamp}\n\n"
            "Only event metadata was sent. No audio left the device.\n"
        )
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout_seconds) as client:
            if self.starttls:
                client.starttls()
            if self.username is not None:
                client.login(self.username, self.password or "")
            client.send_message(message)


class AsyncNotifier(Notifier):
    """Keep network delays and failures off the local safety-alert path."""

    _STOP: Final = object()

    def __init__(self, delegate: Notifier, *, queue_size: int = 32) -> None:
        self.delegate = delegate
        self.errors: list[str] = []
        self.dropped = 0
        self._queue: queue.Queue[AlertNotification | object] = queue.Queue(queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True, name="alert-notifier")
        self._thread.start()

    def notify(self, alert: AlertNotification) -> None:
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            alert = self._queue.get()
            if alert is self._STOP:
                return
            try:
                self.delegate.notify(alert)  # type: ignore[arg-type]
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            return
        self._thread.join(timeout=0.2)
        self.delegate.close()


class NotifyingHardware(HardwareBackend):
    """Decorate a hardware backend with optional, metadata-only notifications."""

    def __init__(self, hardware: HardwareBackend, notifier: Notifier, *, device_name: str) -> None:
        self.hardware = hardware
        self.notifier = notifier
        self.device_name = device_name
        self.last_notification_error: str | None = None

    def check_connection(self) -> None:
        self.hardware.check_connection()

    def set_led(self, enabled: bool) -> None:
        self.hardware.set_led(enabled)

    def set_pwm(self, channel: int, value: float) -> None:
        self.hardware.set_pwm(channel, value)

    def move_servo(self, channel: int, degrees: float) -> None:
        self.hardware.move_servo(channel, degrees)

    def show_alert(self, event: str) -> None:
        self.hardware.show_alert(event)

    def clear_alert(self) -> None:
        self.hardware.clear_alert()

    def apply_decision(self, decision: Decision) -> None:
        self.hardware.apply_decision(decision)
        if decision.notify and decision.event is not None:
            alert = AlertNotification(
                event=decision.event,
                confidence=decision.confidence or 0.0,
                device_name=self.device_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            try:
                self.notifier.notify(alert)
            except Exception as exc:
                self.last_notification_error = f"{type(exc).__name__}: {exc}"

    def shutdown(self) -> None:
        try:
            self.hardware.shutdown()
        finally:
            self.notifier.close()
