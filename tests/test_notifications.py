from edge_ai.decision import Decision
from edge_ai.hardware.mock import MockHardware
from edge_ai.notifications import AlertNotification, Notifier, NotifyingHardware


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
