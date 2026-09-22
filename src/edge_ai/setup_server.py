"""Temporary local web portal for configuring notification preferences."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from html import escape
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
from typing import Final
from urllib.parse import parse_qs, urlsplit

from edge_ai.config import (
    ConfigError,
    build_notification_test_notifier,
    load_notification_setup_config,
)
from edge_ai.notifications import AlertNotification
from edge_ai.settings import (
    NotificationPreferences,
    SettingsError,
    save_notification_preferences,
)


_MAX_FORM_BYTES: Final = 16_384
_COMMON_TIMEZONES: Final = (
    "America/Los_Angeles",
    "America/Denver",
    "America/Chicago",
    "America/New_York",
    "UTC",
)


@dataclass(frozen=True)
class PortalResult:
    message: str
    is_error: bool = False


class SetupPortal:
    """Own the short-lived setup session and its one-time PIN."""

    def __init__(
        self,
        config_path: Path,
        *,
        pin: str | None,
        csrf_token: str | None = None,
        test_sender: Callable[[NotificationPreferences], None] | None = None,
    ) -> None:
        if pin is not None and (not pin.isdigit() or len(pin) != 6):
            raise ValueError("setup PIN must contain exactly six digits")
        configured = load_notification_setup_config(config_path)
        self.config_path = configured.config_path
        self.settings_path = configured.settings_path
        self.preferences = configured.preferences
        self.pin = pin
        self.csrf_token = csrf_token or secrets.token_urlsafe(24)
        self._test_sender = test_sender or self._send_test_email

    def _send_test_email(self, preferences: NotificationPreferences) -> None:
        configured = build_notification_test_notifier(self.config_path, preferences)
        if configured.notifier is None:
            raise RuntimeError("email delivery is not configured")
        alert = AlertNotification(
            event="smoke_alarm",
            confidence=0.95,
            device_name=configured.device_name,
            timestamp=datetime.now(configured.local_timezone).isoformat(timespec="seconds"),
        )
        try:
            configured.notifier.notify(alert)
        except Exception as exc:
            raise RuntimeError(
                f"test email failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            configured.notifier.close()

    def _preferences_from_form(self, form: Mapping[str, str]) -> NotificationPreferences:
        return NotificationPreferences(
            recipient=form.get("recipient", ""),
            device_name=form.get("device_name", ""),
            timezone=form.get("timezone", ""),
            enabled=form.get("enabled") == "on",
        ).validated()

    def submit(self, form: Mapping[str, str]) -> PortalResult:
        if not hmac.compare_digest(form.get("csrf_token", ""), self.csrf_token):
            return PortalResult("This setup page expired. Reload it and try again.", True)
        if self.pin is not None and not hmac.compare_digest(form.get("pin", ""), self.pin):
            return PortalResult("The setup PIN is incorrect.", True)
        try:
            preferences = self._preferences_from_form(form)
            action = form.get("action")
            if action == "test":
                self._test_sender(preferences)
                save_notification_preferences(self.settings_path, preferences)
                self.preferences = preferences
                return PortalResult("Test email delivered. These settings are now saved.")
            if action == "save":
                save_notification_preferences(self.settings_path, preferences)
                self.preferences = preferences
                return PortalResult("Notification settings saved on this device.")
            return PortalResult("Choose Save settings or Save and send test.", True)
        except (ConfigError, RuntimeError, SettingsError) as exc:
            return PortalResult(str(exc), True)

    def render(self, result: PortalResult | None = None) -> bytes:
        preferences = self.preferences
        status = ""
        if result is not None:
            tone = "error" if result.is_error else "success"
            status = f'<div class="status {tone}" role="status">{escape(result.message)}</div>'
        timezone_options = list(_COMMON_TIMEZONES)
        if preferences.timezone not in timezone_options:
            timezone_options.insert(0, preferences.timezone)
        options = "".join(
            f'<option value="{escape(name)}"'
            f'{" selected" if name == preferences.timezone else ""}>{escape(name)}</option>'
            for name in timezone_options
        )
        checked = " checked" if preferences.enabled else ""
        pin_field = ""
        if self.pin is not None:
            pin_field = """
      <label for="pin">Six-digit setup PIN</label>
      <input id="pin" name="pin" type="password" inputmode="numeric" pattern="[0-9]{6}"
        maxlength="6" autocomplete="one-time-code" required>"""
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sound Alert Setup</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --muted:#64717e; --panel:#fff;
      --line:#d9e0e5; --accent:#176b5b; --accent2:#0f5145; --wash:#eef7f4; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; font:16px/1.45 system-ui,-apple-system,sans-serif;
      color:var(--ink); background:linear-gradient(145deg,#edf5f2 0%,#f7f4ec 100%); }}
    main {{ width:min(680px,calc(100% - 32px)); margin:48px auto; }}
    .eyebrow {{ font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
      color:var(--accent); }}
    h1 {{ margin:.35rem 0 .6rem; font-size:clamp(2rem,7vw,3.4rem); line-height:1.02; }}
    .intro {{ color:var(--muted); max-width:58ch; margin-bottom:24px; }}
    .card {{ background:var(--panel); border:1px solid rgba(23,32,42,.09); border-radius:22px;
      box-shadow:0 18px 55px rgba(39,61,55,.12); padding:clamp(22px,5vw,38px); }}
    label {{ display:block; font-weight:750; margin:18px 0 7px; }}
    input,select {{ width:100%; border:1px solid var(--line); border-radius:11px; padding:12px 13px;
      color:var(--ink); background:#fff; font:inherit; }}
    input:focus,select:focus {{ outline:3px solid rgba(23,107,91,.16); border-color:var(--accent); }}
    .hint {{ display:block; color:var(--muted); font-size:.85rem; margin-top:6px; }}
    .toggle {{ display:flex; gap:10px; align-items:center; font-weight:650; margin:20px 0; }}
    .toggle input {{ width:20px; height:20px; margin:0; accent-color:var(--accent); }}
    .actions {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:24px; }}
    button {{ border:0; border-radius:12px; padding:13px 16px; font:inherit; font-weight:800;
      cursor:pointer; background:var(--accent); color:#fff; }}
    button:hover {{ background:var(--accent2); }}
    button.secondary {{ color:var(--accent2); background:var(--wash); }}
    .status {{ padding:12px 14px; margin:0 0 18px; border-radius:11px; font-weight:650; }}
    .success {{ background:#e7f6ee; color:#175b3f; }} .error {{ background:#fff0ed; color:#8b2d1f; }}
    .privacy {{ display:flex; gap:10px; color:var(--muted); font-size:.9rem; margin-top:24px;
      padding-top:20px; border-top:1px solid var(--line); }}
    .dot {{ flex:none; width:10px; height:10px; margin-top:5px; border-radius:50%; background:#20a67a; }}
    @media (max-width:560px) {{ main {{ margin:24px auto; }} .actions {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Private local setup</div>
  <h1>Sound Alert</h1>
  <p class="intro">Choose where safety alerts should be delivered. The microphone audio
    stays on this device; only an event name, time, and device name can be emailed.</p>
  <section class="card">
    {status}
    <form method="post" action="/settings">
      <input type="hidden" name="csrf_token" value="{escape(self.csrf_token)}">
      {pin_field}
      <label for="recipient">Family email addresses</label>
      <input id="recipient" name="recipient" type="email" multiple maxlength="1278"
        value="{escape(preferences.recipient)}" autocomplete="email" required>
      <span class="hint">Enter up to five addresses, separated by commas.</span>
      <label for="device_name">Device or room name</label>
      <input id="device_name" name="device_name" maxlength="80"
        value="{escape(preferences.device_name)}" placeholder="Living room" required>
      <label for="timezone">Timezone</label>
      <select id="timezone" name="timezone">{options}</select>
      <label class="toggle"><input type="checkbox" name="enabled"{checked}>
        Send remote email alerts</label>
      <div class="actions">
        <button class="secondary" type="submit" name="action" value="save">Save settings</button>
        <button type="submit" name="action" value="test">Save and send test</button>
      </div>
    </form>
    <div class="privacy"><span class="dot"></span><span>No recording, waveform, or audio
      feature is included in a notification.</span></div>
  </section>
</main>
</body>
</html>"""
        return page.encode("utf-8")


def _handler_for(portal: SetupPortal, emit: Callable[[str], None]) -> type[BaseHTTPRequestHandler]:
    class SetupRequestHandler(BaseHTTPRequestHandler):
        server_version = "EdgeAISetup/1"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, portal.render(), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._send(200, b"ok\n", "text/plain; charset=utf-8")
            else:
                self._send(404, b"not found\n", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path != "/settings":
                self._send(404, b"not found\n", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > _MAX_FORM_BYTES:
                self._send(413, b"invalid form size\n", "text/plain; charset=utf-8")
                return
            try:
                raw_form = parse_qs(
                    self.rfile.read(length).decode("utf-8", errors="strict"),
                    keep_blank_values=True,
                )
            except UnicodeDecodeError:
                self._send(400, b"form must be UTF-8\n", "text/plain; charset=utf-8")
                return
            form = {key: values[-1] for key, values in raw_form.items()}
            result = portal.submit(form)
            self._send(400 if result.is_error else 200, portal.render(result), "text/html; charset=utf-8")

        def log_message(self, format: str, *args: object) -> None:
            emit(f"setup request: {self.address_string()} {format % args}")

    return SetupRequestHandler


def run_setup_server(
    config_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    emit: Callable[[str], None] = print,
    pin: str | None = None,
) -> None:
    if not 1 <= port <= 65_535:
        raise ValueError("setup port must be between 1 and 65535")
    is_loopback = host in {"127.0.0.1", "localhost", "::1"}
    session_pin = pin if pin is not None else (
        None if is_loopback else f"{secrets.randbelow(1_000_000):06d}"
    )
    portal = SetupPortal(config_path, pin=session_pin)
    server = ThreadingHTTPServer((host, port), _handler_for(portal, emit))
    display_host = host if host not in {"0.0.0.0", "::"} else "<device-ip>"
    emit(f"Notification setup: http://{display_host}:{port}")
    if session_pin is None:
        emit("Setup is restricted to this computer; no PIN is required.")
    else:
        emit(f"One-time setup PIN: {session_pin}")
    emit("Press Ctrl+C when setup is complete; the detector does not need this portal running.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
