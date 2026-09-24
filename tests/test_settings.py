import json
import os
import stat
from pathlib import Path

import pytest

from edge_ai.settings import (
    NotificationPreferences,
    SettingsError,
    load_notification_preferences,
    save_notification_preferences,
)


def test_notification_preferences_round_trip_without_secrets(tmp_path: Path) -> None:
    path = tmp_path / "notification-settings.json"
    preferences = NotificationPreferences(
        recipient="family@example.com",
        device_name="Living room",
        timezone="America/Los_Angeles",
        enabled=True,
    )

    save_notification_preferences(path, preferences)

    assert load_notification_preferences(path) == preferences
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {"version", "recipient", "device_name", "timezone", "enabled"}
    # The UNO Q runs Linux; Windows reports different mode bits for its ACLs.
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("preferences", "message"),
    [
        (NotificationPreferences("not-an-email", "Kitchen", "UTC"), "valid email"),
        (
            NotificationPreferences(
                "one@example.com, two@example.com, three@example.com, four@example.com, "
                "five@example.com, six@example.com",
                "Kitchen",
                "UTC",
            ),
            "one and five",
        ),
        (NotificationPreferences("a@example.com", "", "UTC"), "device name"),
        (NotificationPreferences("a@example.com", "Kitchen", "Mars/Base"), "timezone"),
    ],
)
def test_invalid_notification_preferences_are_rejected(
    preferences: NotificationPreferences, message: str
) -> None:
    with pytest.raises(SettingsError, match=message):
        preferences.validated()


def test_unknown_settings_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notification-settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "recipient": "family@example.com",
                "device_name": "Kitchen",
                "timezone": "UTC",
                "enabled": True,
                "smtp_password": "must-not-be-here",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="unsupported fields"):
        load_notification_preferences(path)


def test_multiple_recipients_are_normalized() -> None:
    preferences = NotificationPreferences(
        "first@example.com,second@example.com", "Kitchen", "UTC"
    ).validated()

    assert preferences.recipient == "first@example.com, second@example.com"
    assert preferences.recipient_addresses == (
        "first@example.com",
        "second@example.com",
    )
