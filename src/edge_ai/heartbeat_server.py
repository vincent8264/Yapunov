"""Watchdog server that emails when a device stops sending heartbeats."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone, tzinfo
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Final
from urllib.parse import urlsplit

from edge_ai.config import HeartbeatServerConfig, load_heartbeat_server_config
from edge_ai.notifications import HealthNotification, Notifier

_MAX_BODY_BYTES: Final = 1024
CHECK_INTERVAL_SECONDS: Final = 1.0


class HeartbeatMonitor:
    """Track one device's heartbeats and report each offline/online transition once."""

    def __init__(
        self,
        *,
        device_name: str,
        missing_after_seconds: float,
        local_timezone: tzinfo = timezone.utc,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[tzinfo], datetime] = datetime.now,
    ) -> None:
        if missing_after_seconds <= 0.0:
            raise ValueError("missing_after_seconds must be positive")
        self.device_name = device_name
        self.missing_after_seconds = missing_after_seconds
        self.local_timezone = local_timezone
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        # Arm at startup so a device that never reports is also detected.
        self._last_seen = clock()
        self._last_seen_at: str | None = None
        self._started_at = self._timestamp()
        self._offline = False

    def _timestamp(self) -> str:
        return self._wall_clock(self.local_timezone).isoformat(timespec="seconds")

    def _notification(self, status: str, reason: str, detail: str) -> HealthNotification:
        return HealthNotification(
            component="device",
            status=status,
            reason=reason,
            detail=detail,
            device_name=self.device_name,
            timestamp=self._timestamp(),
        )

    def record(self) -> HealthNotification | None:
        with self._lock:
            now = self._clock()
            silent_seconds = now - self._last_seen
            self._last_seen = now
            self._last_seen_at = self._timestamp()
            if not self._offline:
                return None
            self._offline = False
            return self._notification(
                "online",
                "heartbeat_received",
                f"heartbeat received after {silent_seconds:.0f} s without one",
            )

    def check(self) -> HealthNotification | None:
        with self._lock:
            silent_seconds = self._clock() - self._last_seen
            if self._offline or silent_seconds < self.missing_after_seconds:
                return None
            self._offline = True
            if self._last_seen_at is None:
                detail = (
                    f"no heartbeat received since the server started at "
                    f"{self._started_at} ({silent_seconds:.0f} s)"
                )
            else:
                detail = (
                    f"no heartbeat for {silent_seconds:.0f} s; last received at "
                    f"{self._last_seen_at}"
                )
            return self._notification("offline", "missed_heartbeat", detail)


def _handler_for(
    monitor: HeartbeatMonitor,
    notifier: Notifier,
    *,
    device_id: str,
    token: str,
    emit: Callable[[str], None],
) -> type[BaseHTTPRequestHandler]:
    expected_authorization = f"Bearer {token}"

    class HeartbeatRequestHandler(BaseHTTPRequestHandler):
        server_version = "EdgeAIHeartbeat/1"

        def _send(self, status: int, body: bytes = b"") -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path == "/healthz":
                self._send(200, b"ok\n")
            else:
                self._send(404, b"not found\n")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path != "/heartbeat":
                self._send(404, b"not found\n")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), expected_authorization
            ):
                # Consume a bounded request body before replying. Otherwise closing
                # the socket with unread POST data can reset the 401 on Windows.
                if 0 < length <= _MAX_BODY_BYTES:
                    self.rfile.read(length)
                self._send(401, b"invalid token\n")
                return
            if length <= 0 or length > _MAX_BODY_BYTES:
                self._send(413, b"invalid body size\n")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, b"body must be UTF-8 JSON\n")
                return
            if not isinstance(payload, dict) or payload.get("device_id") != device_id:
                self._send(404, b"unknown device\n")
                return
            notification = monitor.record()
            if notification is not None:
                emit(f"heartbeat resumed: device={device_id}")
                notifier.notify(notification)
            self._send(204)

        def log_message(self, format: str, *args: object) -> None:
            emit(f"heartbeat request: {self.address_string()} {format % args}")

    return HeartbeatRequestHandler


def watch_heartbeats(
    monitor: HeartbeatMonitor,
    notifier: Notifier,
    stop: threading.Event,
    *,
    device_id: str,
    emit: Callable[[str], None],
    check_interval_seconds: float = CHECK_INTERVAL_SECONDS,
) -> None:
    while not stop.wait(check_interval_seconds):
        notification = monitor.check()
        if notification is not None:
            emit(f"heartbeat missing: device={device_id} {notification.detail}")
            notifier.notify(notification)


def build_heartbeat_server(
    configured: HeartbeatServerConfig,
    monitor: HeartbeatMonitor,
    *,
    emit: Callable[[str], None] = print,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (configured.host, configured.port),
        _handler_for(
            monitor,
            configured.notifier,
            device_id=configured.device_id,
            token=configured.token,
            emit=emit,
        ),
    )


def run_heartbeat_server(config_path: Path, *, emit: Callable[[str], None] = print) -> None:
    configured = load_heartbeat_server_config(config_path)
    monitor = HeartbeatMonitor(
        device_name=configured.device_name,
        missing_after_seconds=configured.missing_after_seconds,
        local_timezone=configured.local_timezone,
    )
    stop = threading.Event()
    try:
        server = build_heartbeat_server(configured, monitor, emit=emit)
    except BaseException:
        configured.notifier.close()
        raise
    watcher = threading.Thread(
        target=watch_heartbeats,
        args=(monitor, configured.notifier, stop),
        kwargs={"device_id": configured.device_id, "emit": emit},
        daemon=True,
        name="heartbeat-watch",
    )
    watcher.start()
    display_host = configured.host if configured.host not in {"0.0.0.0", "::"} else "<server-ip>"
    emit(f"Heartbeat watchdog: http://{display_host}:{configured.port}/heartbeat")
    emit(
        f"Emailing if device {configured.device_id!r} is silent for "
        f"{configured.missing_after_seconds:.0f} s. Press Ctrl+C to stop."
    )
    try:
        server.serve_forever()
    finally:
        stop.set()
        watcher.join(timeout=CHECK_INTERVAL_SECONDS + 1.0)
        server.server_close()
        configured.notifier.close()
