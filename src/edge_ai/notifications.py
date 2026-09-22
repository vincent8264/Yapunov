"""Metadata-only remote notifications for locally classified events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
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


@dataclass(frozen=True)
class NotificationMessage:
    subject: str
    body: str


_EVENT_COPY: Final[dict[str, tuple[str, str, str]]] = {
    "smoke_alarm": (
        "Urgent: smoke alarm sound detected",
        "A smoke alarm sound was detected",
        "Please check immediately.",
    ),
    "glass_break": (
        "Alert: possible glass-breaking sound",
        "A possible glass-breaking sound was detected",
        "Please check the area.",
    ),
    "fall_thud": (
        "Alert: possible fall-like impact",
        "A possible fall-like impact was detected",
        "Please contact the resident.",
    ),
}


def render_notification_message(alert: AlertNotification) -> NotificationMessage:
    """Render fixed, cautious copy without sending event data to a text model."""
    display_event = alert.event.replace("_", " ")
    subject, lead, action = _EVENT_COPY.get(
        alert.event,
        (
            f"Alert: {display_event} sound detected",
            f"A {display_event} sound was detected",
            "Please check the area.",
        ),
    )
    return NotificationMessage(
        subject=f"{subject} at {alert.device_name}",
        body=(
            f"{lead} at {alert.device_name} at {alert.timestamp}.\n\n"
            f"{action}\n\n"
            "Detection details:\n"
            f"Event: {display_event}\n"
            f"Classifier confidence: {alert.confidence:.1%}\n"
            f"Device: {alert.device_name}\n"
            f"Time: {alert.timestamp}\n\n"
            "This is an automated sound-classification alert and does not confirm "
            "an emergency. Only event metadata was sent; no audio left the device.\n"
        ),
    )


class Notifier(ABC):
    channel: str = "notification"

    @abstractmethod
    def notify(self, alert: AlertNotification) -> None: ...

    def close(self) -> None:
        pass


class SMTPNotifier(Notifier):
    """Send event metadata through SMTP; audio is not accepted by this API."""

    channel = "email"

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
        rendered = render_notification_message(alert)
        message = EmailMessage()
        message["Subject"] = rendered.subject
        message["From"] = self.sender
        message["To"] = self.recipient
        message.set_content(rendered.body)
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout_seconds) as client:
            if self.starttls:
                client.starttls()
            if self.username is not None:
                client.login(self.username, self.password or "")
            client.send_message(message)


class AsyncNotifier(Notifier):
    """Keep network delays and failures off the local safety-alert path."""

    _STOP: Final = object()

    def __init__(
        self,
        delegate: Notifier,
        *,
        queue_size: int = 32,
        reporter: Callable[[str], None] | None = None,
        close_timeout_seconds: float = 6.0,
    ) -> None:
        if close_timeout_seconds <= 0.0:
            raise ValueError("notification close timeout must be positive")
        self.delegate = delegate
        self.channel = delegate.channel
        self.reporter = reporter
        self.close_timeout_seconds = close_timeout_seconds
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
            self._report(f"notification dropped: event={alert.event} reason=queue-full")

    def _report(self, message: str) -> None:
        if self.reporter is not None:
            self.reporter(message)

    def _run(self) -> None:
        while True:
            alert = self._queue.get()
            if alert is self._STOP:
                return
            try:
                self.delegate.notify(alert)  # type: ignore[arg-type]
                self._report(
                    f"notification delivered: channel={self.channel} event={alert.event}"
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                self.errors.append(detail)
                self._report(
                    f"notification failed: channel={self.channel} event={alert.event} "
                    f"error={detail}"
                )

    def close(self) -> None:
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            return
        self._thread.join(timeout=self.close_timeout_seconds)
        if self._thread.is_alive():
            self._report("notification shutdown timed out; delivery status is unknown")
        self.delegate.close()


class NotifyingHardware(HardwareBackend):
    """Decorate a hardware backend with optional, metadata-only notifications."""

    def __init__(
        self,
        hardware: HardwareBackend,
        notifier: Notifier,
        *,
        device_name: str,
        local_timezone: tzinfo = timezone.utc,
        clock: Callable[[tzinfo], datetime] = datetime.now,
    ) -> None:
        self.hardware = hardware
        self.notifier = notifier
        self.device_name = device_name
        self.local_timezone = local_timezone
        self.clock = clock
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
                timestamp=self.clock(self.local_timezone).isoformat(timespec="seconds"),
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
