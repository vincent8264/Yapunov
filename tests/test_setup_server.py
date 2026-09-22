from pathlib import Path

from edge_ai.settings import NotificationPreferences, load_notification_preferences
from edge_ai.setup_server import SetupPortal


def _config(path: Path) -> Path:
    config = path / "private.toml"
    config.write_text(
        """
[notifications]
type = "smtp"
settings_file = "settings.json"
device_name = "Home monitor"
timezone = "UTC"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
""",
        encoding="utf-8",
    )
    return config


def _form(portal: SetupPortal, **overrides: str) -> dict[str, str]:
    values = {
        "csrf_token": portal.csrf_token,
        "pin": "123456",
        "recipient": "new-family@example.com",
        "device_name": "Upstairs hallway",
        "timezone": "America/Los_Angeles",
        "enabled": "on",
        "action": "save",
    }
    values.update(overrides)
    return values


def test_setup_portal_saves_valid_nonsecret_preferences(tmp_path: Path) -> None:
    portal = SetupPortal(_config(tmp_path), pin="123456", csrf_token="csrf")

    result = portal.submit(_form(portal))

    assert not result.is_error
    assert load_notification_preferences(tmp_path / "settings.json") == NotificationPreferences(
        "new-family@example.com",
        "Upstairs hallway",
        "America/Los_Angeles",
        True,
    )


def test_setup_portal_requires_pin_and_csrf(tmp_path: Path) -> None:
    portal = SetupPortal(_config(tmp_path), pin="123456", csrf_token="csrf")

    wrong_pin = portal.submit(_form(portal, pin="000000"))
    wrong_csrf = portal.submit(_form(portal, csrf_token="wrong"))

    assert wrong_pin.is_error
    assert "incorrect" in wrong_pin.message
    assert wrong_csrf.is_error
    assert "expired" in wrong_csrf.message
    assert not (tmp_path / "settings.json").exists()


def test_computer_only_setup_does_not_require_pin(tmp_path: Path) -> None:
    portal = SetupPortal(_config(tmp_path), pin=None, csrf_token="csrf")
    form = _form(portal)
    form.pop("pin")

    result = portal.submit(form)
    page = portal.render().decode()

    assert not result.is_error
    assert "Six-digit setup PIN" not in page


def test_setup_portal_test_email_saves_only_after_delivery(tmp_path: Path) -> None:
    delivered: list[NotificationPreferences] = []
    portal = SetupPortal(
        _config(tmp_path),
        pin="123456",
        csrf_token="csrf",
        test_sender=delivered.append,
    )

    result = portal.submit(_form(portal, action="test"))

    assert not result.is_error
    assert delivered == [portal.preferences]
    assert (tmp_path / "settings.json").exists()


def test_setup_portal_rejects_invalid_email_and_escapes_status(tmp_path: Path) -> None:
    portal = SetupPortal(_config(tmp_path), pin="123456", csrf_token="csrf")

    result = portal.submit(_form(portal, recipient="invalid"))
    page = portal.render(type(result)("<script>alert(1)</script>", True)).decode()

    assert result.is_error
    assert not (tmp_path / "settings.json").exists()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_setup_page_does_not_expose_smtp_configuration(tmp_path: Path) -> None:
    page = SetupPortal(
        _config(tmp_path), pin="123456", csrf_token="csrf"
    ).render().decode()

    assert "smtp.example.com" not in page
    assert "EDGE_AI_SMTP_PASSWORD" not in page
    assert "family@example.com" in page
