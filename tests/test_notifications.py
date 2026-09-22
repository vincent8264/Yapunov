from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from edge_ai import notifications
from edge_ai.decision import Decision
from edge_ai.hardware.mock import MockHardware
from edge_ai.notifications import (
    AlertNotification,
    AsyncNotifier,
    Notifier,
    NotifyingHardware,
    SMTPNotifier,
    render_notification_message,
)


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        self.alerts: list[AlertNotification] = []

    def notify(self, alert: AlertNotification) -> None:
        self.alerts.append(alert)


class FailingNotifier(Notifier):
    def notify(self, alert: AlertNotification) -> None:
        raise OSError("network unavailable")


def test_notification_contains_metadata_only_for_new_event() -> None:
    physical = MockHardware(verbose=False)
    notifier = RecordingNotifier()
    hardware = NotifyingHardware(physical, notifier, device_name="Kitchen")

    hardware.apply_decision(
        Decision("alert", event="smoke_alarm", confidence=0.93, notify=True)
    )
    hardware.apply_decision(
        Decision("alert", event="smoke_alarm", confidence=0.94, notify=False)
    )

    assert physical.current_alert == "smoke_alarm"
    assert len(notifier.alerts) == 1
    assert notifier.alerts[0].device_name == "Kitchen"
    assert set(notifier.alerts[0].__dataclass_fields__) == {
        "event",
        "confidence",
        "device_name",
        "timestamp",
    }


def test_notification_failure_does_not_cancel_local_alert() -> None:
    physical = MockHardware(verbose=False)
    hardware = NotifyingHardware(physical, FailingNotifier(), device_name="Bedroom")

    hardware.apply_decision(
        Decision("alert", event="glass_break", confidence=0.9, notify=True)
    )

    assert physical.current_alert == "glass_break"
    assert hardware.last_notification_error == "OSError: network unavailable"


def test_fixed_messages_use_cautious_event_specific_copy() -> None:
    glass = render_notification_message(
        AlertNotification(
            event="glass_break",
            confidence=0.91,
            device_name="Living room",
            timestamp="2026-09-22T14:30:00-07:00",
        )
    )
    fall = render_notification_message(
        AlertNotification(
            event="fall_thud",
            confidence=0.88,
            device_name="Living room",
            timestamp="2026-09-22T14:31:00-07:00",
        )
    )
    smoke = render_notification_message(
        AlertNotification(
            event="smoke_alarm",
            confidence=0.97,
            device_name="Living room",
            timestamp="2026-09-22T14:32:00-07:00",
        )
    )

    assert "possible glass" in glass.subject.lower()
    assert "does not confirm an emergency" in glass.body
    assert "possible fall" in fall.subject.lower()
    assert "contact the resident" in fall.body
    assert smoke.subject.startswith("Urgent:")
    assert "check immediately" in smoke.body
    assert "no audio left the device" in smoke.body


def test_smtp_notifier_sends_rendered_metadata_only_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = []

    class FakeSMTP:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            assert (host, port, timeout) == ("smtp.example.com", 587, 5.0)

        def __enter__(self) -> "FakeSMTP":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def starttls(self) -> None:
            pass

        def login(self, username: str, password: str) -> None:
            assert (username, password) == ("monitor@example.com", "secret")

        def send_message(self, message: object) -> None:
            sent.append(message)

    monkeypatch.setattr(notifications.smtplib, "SMTP", FakeSMTP)
    notifier = SMTPNotifier(
        host="smtp.example.com",
        port=587,
        sender="monitor@example.com",
        recipient="family@example.com",
        username="monitor@example.com",
        password="secret",
    )

    notifier.notify(
        AlertNotification(
            event="fall_thud",
            confidence=0.88,
            device_name="Living room",
            timestamp="2026-09-22T14:31:00-07:00",
        )
    )

    assert len(sent) == 1
    message = sent[0]
    assert "possible fall-like impact" in str(message["Subject"]).lower()
    assert "88.0%" in message.get_content()
    assert "no audio left the device" in message.get_content()


def test_notification_timestamp_uses_configured_timezone() -> None:
    physical = MockHardware(verbose=False)
    notifier = RecordingNotifier()
    local_timezone = ZoneInfo("America/Los_Angeles")
    fixed_time = datetime(2026, 9, 22, 14, 30, tzinfo=local_timezone)
    hardware = NotifyingHardware(
        physical,
        notifier,
        device_name="Kitchen",
        local_timezone=local_timezone,
        clock=lambda timezone: fixed_time.astimezone(timezone),
    )

    hardware.apply_decision(
        Decision("alert", event="smoke_alarm", confidence=0.93, notify=True)
    )

    assert notifier.alerts[0].timestamp == "2026-09-22T14:30:00-07:00"


def test_async_notifier_reports_delivery_and_failure() -> None:
    alert = AlertNotification("glass_break", 0.9, "Kitchen", "2026-09-22T14:30:00-07:00")
    delivered: list[str] = []
    successful = AsyncNotifier(RecordingNotifier(), reporter=delivered.append)
    successful.notify(alert)
    successful.close()

    failed: list[str] = []
    failing = AsyncNotifier(FailingNotifier(), reporter=failed.append)
    failing.notify(alert)
    failing.close()

    assert delivered == ["notification delivered: channel=notification event=glass_break"]
    assert len(failed) == 1
    assert failed[0].startswith("notification failed: channel=notification event=glass_break")
