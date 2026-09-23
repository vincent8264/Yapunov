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
class HealthNotification:
    component: str
    status: str
    reason: str
    detail: str
    device_name: str
    timestamp: str


Notification = AlertNotification | HealthNotification


def _notification_log_field(notification: Notification) -> str:
    if isinstance(notification, HealthNotification):
        return f"health={notification.component}_{notification.status}"
    return f"event={notification.event}"


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
    "help_call": (
        "Urgent: possible call for help detected",
        "A possible spoken call for help was detected",
        "Please contact the resident immediately.",
    ),
}


def render_notification_message(alert: Notification) -> NotificationMessage:
    """Render fixed, cautious copy without sending event data to a text model."""
    if isinstance(alert, HealthNotification):
        if alert.status == "fault":
            return NotificationMessage(
                subject=f"Device warning: microphone unavailable at {alert.device_name}",
                body=(
                    f"The microphone input at {alert.device_name} became unavailable "
                    f"at {alert.timestamp}.\n\n"
                    f"Observed condition: {alert.reason.replace('_', ' ')}\n"
                    f"Detail: {alert.detail}\n"
                    f"Device: {alert.device_name}\n\n"
                    "The device is attempting local recovery. This observation does "
                    "not prove that the microphone hardware is broken. Only device "
                    "health metadata was sent; no audio left the device.\n"
                ),
            )
        return NotificationMessage(
            subject=f"Device recovered: microphone available at {alert.device_name}",
            body=(
                f"The microphone input at {alert.device_name} resumed at "
                f"{alert.timestamp}.\n\n"
                f"Previous condition: {alert.reason.replace('_', ' ')}\n"
                f"Device: {alert.device_name}\n\n"
                "Only device health metadata was sent; no audio left the device.\n"
            ),
        )
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
    # When True, notify() only queues the alert; results come from drain_delivery_results().
    delivers_later: bool = False

    @abstractmethod
    def notify(self, alert: Notification) -> None: ...

    def drain_delivery_results(self) -> list[tuple[Notification, bool]]:
        return []

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

    def notify(self, alert: Notification) -> None:
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
    delivers_later = True

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
        self._queue: queue.Queue[Notification | object] = queue.Queue(queue_size)
        self._results: queue.SimpleQueue[tuple[Notification, bool]] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="alert-notifier")
        self._thread.start()

    def notify(self, alert: Notification) -> None:
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            self.dropped += 1
            self._results.put((alert, False))
            self._report(
                f"notification dropped: {_notification_log_field(alert)} reason=queue-full"
            )

    def drain_delivery_results(self) -> list[tuple[Notification, bool]]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

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
                self._results.put((alert, True))  # type: ignore[arg-type]
                self._report(
                    f"notification delivered: channel={self.channel} "
                    f"{_notification_log_field(alert)}"
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                self.errors.append(detail)
                self._results.put((alert, False))  # type: ignore[arg-type]
                self._report(
                    f"notification failed: channel={self.channel} "
                    f"{_notification_log_field(alert)} "
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
        self._displayed_notification: Notification | None = None
        self._input_fault: tuple[str, str] | None = None

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

    def show_spectrum(self, columns: tuple[int, ...]) -> None:
        self._forward_delivery_results()
        self.hardware.show_spectrum(columns)

    def show_notification_status(self, delivered: bool) -> None:
        self.hardware.show_notification_status(delivered)

    def refresh_status(self) -> None:
        self._forward_delivery_results()
        self.hardware.refresh_status()

    def _forward_delivery_results(self) -> None:
        # Results for alerts that are no longer on screen must not light the new one.
        for alert, delivered in self.notifier.drain_delivery_results():
            if alert is self._displayed_notification:
                self.hardware.show_notification_status(delivered)

    def _notify(self, notification: Notification, *, displayed: bool) -> None:
        if displayed:
            self._displayed_notification = notification
        delivered = False
        try:
            self.notifier.notify(notification)
            delivered = not self.notifier.delivers_later
        except Exception as exc:
            self.last_notification_error = f"{type(exc).__name__}: {exc}"
        if displayed:
            self.hardware.show_notification_status(delivered)

    def show_input_fault(self, reason: str, detail: str) -> None:
        if self._input_fault is not None:
            self.refresh_status()
            return
        self._forward_delivery_results()
        self.hardware.show_input_fault(reason, detail)
        self._input_fault = (reason, detail)
        notification = HealthNotification(
            component="microphone",
            status="fault",
            reason=reason,
            detail=detail,
            device_name=self.device_name,
            timestamp=self.clock(self.local_timezone).isoformat(timespec="seconds"),
        )
        self._notify(notification, displayed=True)

    def clear_input_fault(self, reason: str) -> None:
        if self._input_fault is None:
            return
        self._forward_delivery_results()
        self.hardware.clear_input_fault(reason)
        previous = self._input_fault
        self._input_fault = None
        self._displayed_notification = None
        if previous is None:
            return
        notification = HealthNotification(
            component="microphone",
            status="recovered",
            reason=previous[0],
            detail=f"microphone input resumed after {previous[0].replace('_', ' ')}",
            device_name=self.device_name,
            timestamp=self.clock(self.local_timezone).isoformat(timespec="seconds"),
        )
        self._notify(notification, displayed=False)

    def apply_decision(self, decision: Decision) -> None:
        self._forward_delivery_results()
        self.hardware.apply_decision(decision)
        if (
            isinstance(self._displayed_notification, AlertNotification)
            and (
                decision.action != "alert"
                or decision.event != self._displayed_notification.event
            )
        ):
            self._displayed_notification = None
        if decision.notify and decision.event is not None:
            alert = AlertNotification(
                event=decision.event,
                confidence=decision.confidence or 0.0,
                device_name=self.device_name,
                timestamp=self.clock(self.local_timezone).isoformat(timespec="seconds"),
            )
            self._notify(alert, displayed=True)

    def shutdown(self) -> None:
        try:
            self.hardware.shutdown()
        finally:
            self.notifier.close()
