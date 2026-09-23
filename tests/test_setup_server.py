from collections.abc import Callable
from pathlib import Path

import pytest

from edge_ai import setup_server
from edge_ai.settings import NotificationPreferences, load_notification_preferences
from edge_ai.setup_server import SetupPortal, _persistent_pin


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


def test_setup_portal_notifies_running_detector_after_save(tmp_path: Path) -> None:
    reloaded: list[NotificationPreferences] = []
    portal = SetupPortal(
        _config(tmp_path),
        pin="123456",
        csrf_token="csrf",
        on_saved=reloaded.append,
    )

    result = portal.submit(_form(portal))

    assert not result.is_error
    assert reloaded == [portal.preferences]


def test_persistent_setup_pin_is_reused_and_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "private" / "setup-pin"

    first = _persistent_pin(path)
    second = _persistent_pin(path)

    assert first == second
    assert first.isdigit() and len(first) == 6
    assert path.stat().st_mode & 0o777 == 0o600


def test_configured_background_server_starts_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        + """
[setup]
enabled = true
host = "0.0.0.0"
port = 8123
pin_file = "setup-pin"
""",
        encoding="utf-8",
    )
    calls: list[object] = []

    class FakeServer:
        daemon_threads = False

        def __init__(self, address: tuple[str, int], handler: object) -> None:
            calls.append((address, handler))

        def serve_forever(self) -> None:
            calls.append("served")

        def shutdown(self) -> None:
            calls.append("shutdown")

        def server_close(self) -> None:
            calls.append("closed")

    class FakeThread:
        def __init__(self, *, target: Callable[[], None], **_: object) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

        def join(self, *, timeout: float) -> None:
            calls.append(("joined", timeout))

    monkeypatch.setattr(setup_server, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(setup_server.threading, "Thread", FakeThread)
    output: list[str] = []

    running = setup_server.start_configured_setup_server(config, emit=output.append)
    assert running is not None
    running.close()

    first_call = calls[0]
    assert isinstance(first_call, tuple)
    assert first_call[0] == ("0.0.0.0", 8123)
    assert calls[1:] == ["served", "shutdown", "closed", ("joined", 2.0)]
    assert any("http://<device-ip>:8123" in line for line in output)
    assert any(line.startswith("Setup PIN: ") for line in output)
    assert (tmp_path / "setup-pin").is_file()


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
