from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from edge_ai import notifications
from edge_ai.decision import Decision, RiskWarning
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
        "detail",
    }
    assert notifier.alerts[0].detail is None


def _water_warning() -> RiskWarning:
    return RiskWarning(
        risk="water_leak",
        message="Possible water leak detected",
        detected_seconds=7.25,
        window_seconds=15.0,
        peak_score=0.42,
        peak_label="Water tap, faucet",
    )


def test_risk_warning_sends_one_email_without_changing_the_matrix() -> None:
    physical = MockHardware(verbose=False)
    notifier = RecordingNotifier()
    hardware = NotifyingHardware(physical, notifier, device_name="Kitchen")

    hardware.apply_decision(Decision("idle", warnings=(_water_warning(),)))
    hardware.apply_decision(Decision("idle"))

    assert len(notifier.alerts) == 1
    alert = notifier.alerts[0]
    assert alert.event == "water_leak"
    assert alert.confidence == pytest.approx(0.42)
    assert alert.detail == (
        "detected for 7 s of the last 15 s (strongest sound: Water tap, faucet)"
    )
    assert physical.current_alert is None
    assert physical.notification_delivered is False


def test_risk_warning_email_accompanies_an_emergency_email() -> None:
    physical = MockHardware(verbose=False)
    notifier = RecordingNotifier()
    hardware = NotifyingHardware(physical, notifier, device_name="Kitchen")

    hardware.apply_decision(
        Decision(
            "alert",
            event="smoke_alarm",
            confidence=0.9,
            notify=True,
            warnings=(_water_warning(),),
        )
    )

    assert [alert.event for alert in notifier.alerts] == ["smoke_alarm", "water_leak"]
    assert physical.current_alert == "smoke_alarm"
    assert physical.notification_delivered is True


def test_risk_warning_email_failure_is_recorded_not_raised() -> None:
    hardware = NotifyingHardware(
        MockHardware(verbose=False), FailingNotifier(), device_name="Kitchen"
    )

    hardware.apply_decision(Decision("idle", warnings=(_water_warning(),)))

    assert hardware.last_notification_error == "OSError: network unavailable"


def test_risk_warning_messages_use_cautious_copy() -> None:
    water = render_notification_message(
        AlertNotification(
            event="water_leak",
            confidence=0.42,
            device_name="Kitchen",
            timestamp="2026-09-22T14:30:00-07:00",
            detail="detected for 7 s of the last 15 s (strongest sound: Drip)",
        )
    )
    cooking = render_notification_message(
        AlertNotification(
            event="unattended_cooking",
            confidence=0.5,
            device_name="Kitchen",
            timestamp="2026-09-22T14:30:00-07:00",
        )
    )

    assert water.subject == "Warning: possible water leak at Kitchen"
    assert "Pattern: detected for 7 s of the last 15 s (strongest sound: Drip)" in water.body
    assert "does not confirm an emergency" in water.body
    assert cooking.subject == "Warning: possible unattended cooking at Kitchen"
    assert "stove is attended" in cooking.body
    assert "Pattern:" not in cooking.body


def test_notification_failure_does_not_cancel_local_alert() -> None:
    physical = MockHardware(verbose=False)
    hardware = NotifyingHardware(physical, FailingNotifier(), device_name="Bedroom")

    hardware.apply_decision(
        Decision("alert", event="glass_break", confidence=0.9, notify=True)
    )

    assert physical.current_alert == "glass_break"
    assert hardware.last_notification_error == "OSError: network unavailable"


def test_synchronous_delivery_lights_status_and_failure_leaves_it_off() -> None:
    delivered_hw = MockHardware(verbose=False)
    NotifyingHardware(delivered_hw, RecordingNotifier(), device_name="Kitchen").apply_decision(
        Decision("alert", event="smoke_alarm", confidence=0.9, notify=True)
    )
    failed_hw = MockHardware(verbose=False)
    NotifyingHardware(failed_hw, FailingNotifier(), device_name="Kitchen").apply_decision(
        Decision("alert", event="smoke_alarm", confidence=0.9, notify=True)
    )

    assert delivered_hw.notification_delivered is True
    assert failed_hw.notification_delivered is False


class QueuedNotifier(Notifier):
    """Deterministic stand-in for AsyncNotifier: results are released by the test."""

    delivers_later = True

    def __init__(self) -> None:
        self.alerts: list[AlertNotification] = []
        self.results: list[tuple[AlertNotification, bool]] = []

    def notify(self, alert: AlertNotification) -> None:
        self.alerts.append(alert)

    def drain_delivery_results(self) -> list[tuple[AlertNotification, bool]]:
        results, self.results = self.results, []
        return results


def test_later_delivery_updates_status_on_next_main_thread_call() -> None:
    physical = MockHardware(verbose=False)
    notifier = QueuedNotifier()
    hardware = NotifyingHardware(physical, notifier, device_name="Kitchen")

    hardware.apply_decision(Decision("alert", event="glass_break", confidence=0.9, notify=True))
    assert physical.notification_delivered is False

    notifier.results.append((notifier.alerts[0], True))
    hardware.show_spectrum((0,) * 13)

    assert physical.notification_delivered is True


def test_stale_delivery_result_does_not_light_a_later_alert() -> None:
    physical = MockHardware(verbose=False)
    notifier = QueuedNotifier()
    hardware = NotifyingHardware(physical, notifier, device_name="Kitchen")

    hardware.apply_decision(Decision("alert", event="glass_break", confidence=0.9, notify=True))
    hardware.apply_decision(Decision("idle"))
    hardware.apply_decision(Decision("alert", event="fall_thud", confidence=0.9, notify=False))
    notifier.results.append((notifier.alerts[0], True))
    hardware.show_spectrum((0,) * 13)

    assert physical.current_alert == "fall_thud"
    assert physical.notification_delivered is False


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
        recipient="family@example.com, neighbor@example.com",
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
    assert str(message["To"]) == "family@example.com, neighbor@example.com"
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
    assert successful.drain_delivery_results() == [(alert, True)]
    assert failing.drain_delivery_results() == [(alert, False)]
    assert successful.drain_delivery_results() == []
