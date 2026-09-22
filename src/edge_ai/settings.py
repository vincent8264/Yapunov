"""Validated, metadata-only notification preferences stored on the local device."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class SettingsError(ValueError):
    """Raised when saved notification preferences are missing or invalid."""


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class NotificationPreferences:
    recipient: str
    device_name: str
    timezone: str
    enabled: bool = True

    @property
    def recipient_addresses(self) -> tuple[str, ...]:
        addresses = tuple(address.strip() for address in self.recipient.split(","))
        if not 1 <= len(addresses) <= 5:
            raise SettingsError("enter between one and five recipient email addresses")
        if any(
            not address
            or not _EMAIL_PATTERN.fullmatch(address)
            or "\r" in address
            or "\n" in address
            for address in addresses
        ):
            raise SettingsError("recipients must be valid email addresses separated by commas")
        if len({address.casefold() for address in addresses}) != len(addresses):
            raise SettingsError("recipient email addresses must not be duplicated")
        return addresses

    def validated(self) -> "NotificationPreferences":
        if not all(
            isinstance(value, str)
            for value in (self.recipient, self.device_name, self.timezone)
        ):
            raise SettingsError("recipient, device name, and timezone must be strings")
        recipient = ", ".join(self.recipient_addresses)
        device_name = self.device_name.strip()
        timezone = self.timezone.strip()
        if not device_name or len(device_name) > 80 or "\r" in device_name or "\n" in device_name:
            raise SettingsError("device name must contain 1 to 80 characters")
        if not timezone:
            raise SettingsError("timezone must be a non-empty IANA timezone name")
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise SettingsError(f"unknown timezone: {timezone!r}") from exc
        if not isinstance(self.enabled, bool):
            raise SettingsError("enabled must be true or false")
        return NotificationPreferences(recipient, device_name, timezone, self.enabled)


def load_notification_preferences(path: Path) -> NotificationPreferences:
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SettingsError(f"notification settings file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SettingsError(f"could not read notification settings file {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("version") != 1:
        raise SettingsError("notification settings must use schema version 1")
    expected = {"version", "recipient", "device_name", "timezone", "enabled"}
    if set(document) != expected:
        raise SettingsError("notification settings contain missing or unsupported fields")
    try:
        preferences = NotificationPreferences(
            recipient=document["recipient"],
            device_name=document["device_name"],
            timezone=document["timezone"],
            enabled=document["enabled"],
        )
    except TypeError as exc:
        raise SettingsError("notification settings fields have invalid types") from exc
    return preferences.validated()


def save_notification_preferences(path: Path, preferences: NotificationPreferences) -> None:
    """Atomically save non-secret preferences with owner-only file permissions."""
    validated = preferences.validated()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, **asdict(validated)}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise SettingsError(f"could not save notification settings to {path}: {exc}") from exc
