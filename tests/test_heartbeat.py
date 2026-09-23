from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from edge_ai.config import (
    ConfigError,
    ConfiguredPipeline,
    HeartbeatServerConfig,
    load_config,
    load_heartbeat_server_config,
)
from edge_ai.decision import decide
from edge_ai.hardware.mock import MockHardware
from edge_ai.heartbeat import HeartbeatSender, post_heartbeat
from edge_ai.heartbeat_server import HeartbeatMonitor, build_heartbeat_server
from edge_ai.inference.dummy import DummyInferenceEngine
from edge_ai.inputs.base import InputSource
from edge_ai.notifications import HealthNotification, Notifier, render_notification_message
from edge_ai.pipeline import Pipeline
from edge_ai.runner import run_pipeline


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        self.notifications: list[HealthNotification] = []

    def notify(self, alert: HealthNotification) -> None:  # type: ignore[override]
        self.notifications.append(alert)


def _fixed_wall(tz: object) -> datetime:
    return datetime(2026, 9, 23, 9, 0, tzinfo=timezone.utc)


def _sender(clock: FakeClock, post, reports: list[str] | None = None) -> HeartbeatSender:
    return HeartbeatSender(
        url="http://192.0.2.10:8090/heartbeat",
        device_id="home-monitor",
        token="secret-token",
        interval_seconds=60.0,
        stall_seconds=30.0,
        reporter=None if reports is None else reports.append,
        post=post,
        clock=clock,
    )


def test_sender_posts_device_id_with_bearer_token() -> None:
    calls = []
    sender = _sender(FakeClock(), lambda *args: calls.append(args))

    assert sender.send_once() is True

    url, body, headers, timeout = calls[0]
    assert url == "http://192.0.2.10:8090/heartbeat"
    assert json.loads(body) == {"device_id": "home-monitor"}
    assert headers["Authorization"] == "Bearer secret-token"
    assert timeout == 5.0


def test_sender_pauses_when_main_loop_stalls_and_resumes_after_beat() -> None:
    clock = FakeClock()
    calls = []
    reports: list[str] = []
    sender = _sender(clock, lambda *args: calls.append(args), reports)

    clock.now += 31.0
    assert sender.send_once() is False
    assert sender.send_once() is False
    sender.beat()
    assert sender.send_once() is True

    assert len(calls) == 1
    assert reports == ["heartbeat paused: main loop idle for 31 s"]


def test_sender_reports_network_failure_once_and_never_raises() -> None:
    failing = True
    reports: list[str] = []

    def post(*args: object) -> None:
        if failing:
            raise OSError("network unreachable")

    sender = _sender(FakeClock(), post, reports)

    assert sender.send_once() is False
    assert sender.send_once() is False
    failing = False
    assert sender.send_once() is True
    assert reports == [
        "heartbeat failed: OSError: network unreachable",
        "heartbeat restored",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"url": "192.0.2.10/heartbeat"}, "http"),
        ({"device_id": "has space"}, "device_id"),
        ({"token": ""}, "token"),
        ({"interval_seconds": 0.0}, "interval"),
        ({"timeout_seconds": 60.0}, "timeout"),
        ({"stall_seconds": 0.0}, "stall"),
    ],
)
def test_sender_rejects_invalid_settings(overrides: dict, message: str) -> None:
    settings = {"url": "http://192.0.2.10/heartbeat", "device_id": "dev", "token": "t"}
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        HeartbeatSender(**settings)


def test_sender_thread_sends_immediately_and_stops_on_close() -> None:
    sent = threading.Event()
    sender = HeartbeatSender(
        url="http://192.0.2.10/heartbeat",
        device_id="dev",
        token="t",
        interval_seconds=60.0,
        post=lambda *args: sent.set(),
    )
    sender.start()
    assert sent.wait(2.0)
    sender.close()
    assert sender._thread is not None and not sender._thread.is_alive()


def _monitor(clock: FakeClock) -> HeartbeatMonitor:
    return HeartbeatMonitor(
        device_name="Kitchen",
        missing_after_seconds=150.0,
        clock=clock,
        wall_clock=_fixed_wall,
    )


def test_monitor_reports_missing_device_once_then_recovery() -> None:
    clock = FakeClock()
    monitor = _monitor(clock)
    assert monitor.record() is None

    clock.now += 149.0
    assert monitor.check() is None
    clock.now += 1.0
    offline = monitor.check()
    clock.now += 60.0
    assert monitor.check() is None
    online = monitor.record()

    assert offline is not None and offline.status == "offline"
    assert offline.component == "device"
    assert "no heartbeat for 150 s" in offline.detail
    assert online is not None and online.status == "online"
    assert "after 210 s" in online.detail
    assert monitor.record() is None


def test_monitor_reports_device_that_never_sends_after_server_start() -> None:
    clock = FakeClock()
    monitor = _monitor(clock)

    clock.now += 150.0
    offline = monitor.check()

    assert offline is not None
    assert "since the server started" in offline.detail


def test_heartbeat_emails_use_device_wording() -> None:
    notification = HealthNotification(
        component="device",
        status="offline",
        reason="missed_heartbeat",
        detail="no heartbeat for 150 s",
        device_name="Kitchen",
        timestamp="2026-09-23T09:00:00+00:00",
    )
    offline = render_notification_message(notification)
    online = render_notification_message(
        HealthNotification(**{**notification.__dict__, "status": "online"})
    )

    assert offline.subject == "Device warning: no heartbeat from Kitchen"
    assert "local sound alerts may not be working" in offline.body
    assert "microphone" not in offline.body
    assert online.subject == "Device recovered: heartbeat resumed from Kitchen"


@pytest.fixture
def live_server():
    clock = FakeClock()
    monitor = _monitor(clock)
    notifier = RecordingNotifier()
    configured = HeartbeatServerConfig(
        host="127.0.0.1",
        port=0,
        device_id="home-monitor",
        token="secret-token",
        missing_after_seconds=150.0,
        notifier=notifier,
        device_name="Kitchen",
        local_timezone=timezone.utc,
    )
    server = build_heartbeat_server(configured, monitor, emit=lambda message: None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/heartbeat"
    yield url, clock, monitor, notifier
    server.shutdown()
    server.server_close()


def _post(url: str, body: bytes, token: str = "secret-token") -> int:
    request = Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            return response.status
    except HTTPError as exc:
        return exc.code


def test_server_accepts_sender_heartbeat_and_emails_recovery(live_server) -> None:
    url, clock, monitor, notifier = live_server
    clock.now += 200.0
    assert monitor.check() is not None

    sender = HeartbeatSender(
        url=url, device_id="home-monitor", token="secret-token", post=post_heartbeat
    )
    assert sender.send_once() is True

    assert [n.status for n in notifier.notifications] == ["online"]


@pytest.mark.parametrize(
    ("body", "token", "status"),
    [
        (b'{"device_id": "home-monitor"}', "wrong-token", 401),
        (b'{"device_id": "other-device"}', "secret-token", 404),
        (b"not json", "secret-token", 400),
        (b"x" * 2048, "secret-token", 413),
    ],
)
def test_server_rejects_invalid_heartbeats(live_server, body, token, status) -> None:
    url, clock, monitor, notifier = live_server
    clock.now += 200.0

    assert _post(url, body, token) == status
    assert monitor.check() is not None
    assert notifier.notifications == []


_PIPELINE = """
[runtime]
[input]
type = "simulated_sensor"
seed = 1
[preprocessing]
type = "identity"
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
verbose = false
"""

_HEARTBEAT = """
[heartbeat]
url = "http://192.0.2.10:8090/heartbeat"
device_id = "home-monitor"
token_file = "token.txt"
interval_seconds = 30.0
"""

_SERVER = """
[notifications]
type = "smtp"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
device_name = "Kitchen"
[heartbeat_server]
port = 8091
missing_after_seconds = 75.0
"""


def test_config_builds_heartbeat_sender_from_token_file(tmp_path: Path) -> None:
    (tmp_path / "token.txt").write_text("file-token\n", encoding="utf-8")
    path = tmp_path / "live.toml"
    path.write_text(_PIPELINE + _HEARTBEAT, encoding="utf-8")

    heartbeat = load_config(path).heartbeat

    assert heartbeat is not None
    assert heartbeat.device_id == "home-monitor"
    assert heartbeat.interval_seconds == 30.0
    assert heartbeat._token == "file-token"


def test_config_without_heartbeat_section_has_no_sender(tmp_path: Path) -> None:
    path = tmp_path / "live.toml"
    path.write_text(_PIPELINE, encoding="utf-8")

    assert load_config(path).heartbeat is None


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('token_file = "missing.txt"', "token file not found"),
        ('token_env = "EDGE_AI_TEST_UNSET_TOKEN"', "environment variable is not set"),
        ('token_file = "token.txt"\ntoken_env = "X"', "exactly one"),
    ],
)
def test_config_rejects_missing_heartbeat_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str, message: str
) -> None:
    monkeypatch.delenv("EDGE_AI_TEST_UNSET_TOKEN", raising=False)
    (tmp_path / "token.txt").write_text("file-token\n", encoding="utf-8")
    path = tmp_path / "live.toml"
    path.write_text(
        _PIPELINE + _HEARTBEAT.replace('token_file = "token.txt"', replacement),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_config_rejects_invalid_heartbeat_url(tmp_path: Path) -> None:
    (tmp_path / "token.txt").write_text("file-token\n", encoding="utf-8")
    path = tmp_path / "live.toml"
    path.write_text(
        _PIPELINE + _HEARTBEAT.replace("http://192.0.2.10:8090", "ftp://example"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"invalid \[heartbeat\].*http"):
        load_config(path)


def test_server_config_shares_heartbeat_identity_and_smtp(tmp_path: Path) -> None:
    (tmp_path / "token.txt").write_text("file-token\n", encoding="utf-8")
    path = tmp_path / "private.toml"
    path.write_text(_HEARTBEAT + _SERVER, encoding="utf-8")

    configured = load_heartbeat_server_config(path)
    configured.notifier.close()

    assert configured.device_id == "home-monitor"
    assert configured.token == "file-token"
    assert "file-token" not in repr(configured)
    assert configured.host == "127.0.0.1"
    assert configured.port == 8091
    assert configured.missing_after_seconds == 75.0
    assert configured.device_name == "Kitchen"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_HEARTBEAT + _SERVER.replace("75.0", "30.0"), "longer than"),
        (_HEARTBEAT + _SERVER.replace("8091", "0"), "port"),
        (_HEARTBEAT + _SERVER.replace('type = "smtp"', 'type = "none"'), "requires enabled"),
        (_SERVER, r"\[heartbeat\]"),
        (_HEARTBEAT, r"\[heartbeat_server\]"),
    ],
)
def test_server_config_rejects_invalid_settings(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "token.txt").write_text("file-token\n", encoding="utf-8")
    path = tmp_path / "private.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_heartbeat_server_config(path)


class OneValueInput(InputSource):
    def read(self) -> float:
        return 1.0


class RecordingHeartbeat:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def beat(self) -> None:
        self.events.append("beat")

    def close(self) -> None:
        self.events.append("close")


def test_runner_starts_beats_and_closes_heartbeat() -> None:
    heartbeat = RecordingHeartbeat()
    pipeline = Pipeline(
        OneValueInput(),
        lambda value: value,
        DummyInferenceEngine(),
        decide,
        MockHardware(verbose=False),
    )
    configured = ConfiguredPipeline(pipeline, 0.0, heartbeat=heartbeat)  # type: ignore[arg-type]

    run_pipeline(configured, max_steps=2, emit=lambda message: None)

    assert heartbeat.events == ["start", "beat", "beat", "close"]
