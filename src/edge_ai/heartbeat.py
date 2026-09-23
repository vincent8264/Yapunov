"""Periodic liveness heartbeats from the device to an external watchdog server."""

from __future__ import annotations

from collections.abc import Callable
import json
import re
import threading
import time
from typing import Final
from urllib.request import Request, urlopen

DEVICE_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9._-]{1,64}")

HeartbeatPoster = Callable[[str, bytes, dict[str, str], float], None]


def post_heartbeat(url: str, body: bytes, headers: dict[str, str], timeout: float) -> None:
    """POST one heartbeat; any HTTP error status raises."""
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout):
        pass


class HeartbeatSender:
    """Send a heartbeat every ``interval_seconds`` while the main loop is progressing.

    The main loop calls :meth:`beat` on every iteration. If it has not done so for
    ``stall_seconds``, heartbeats stop so the watchdog also detects a hung pipeline,
    not only a powered-off or disconnected board. Network failures never propagate
    to the caller.
    """

    def __init__(
        self,
        *,
        url: str,
        device_id: str,
        token: str,
        interval_seconds: float = 60.0,
        timeout_seconds: float = 5.0,
        stall_seconds: float = 30.0,
        reporter: Callable[[str], None] | None = None,
        post: HeartbeatPoster = post_heartbeat,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("heartbeat url must start with http:// or https://")
        if not DEVICE_ID_PATTERN.fullmatch(device_id):
            raise ValueError(
                "heartbeat device_id must be 1-64 letters, digits, '.', '_', or '-'"
            )
        if not token:
            raise ValueError("heartbeat token must not be empty")
        if interval_seconds <= 0.0:
            raise ValueError("heartbeat interval must be positive")
        if not 0.0 < timeout_seconds < interval_seconds:
            raise ValueError("heartbeat timeout must be positive and shorter than the interval")
        if stall_seconds <= 0.0:
            raise ValueError("heartbeat stall time must be positive")
        self.url = url
        self.device_id = device_id
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.stall_seconds = stall_seconds
        self.reporter = reporter
        self._token = token
        self._post = post
        self._clock = clock
        self._last_progress = clock()
        self._failing = False
        self._stalled = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._last_progress = self._clock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
        self._thread.start()

    def beat(self) -> None:
        self._last_progress = self._clock()

    def send_once(self) -> bool:
        idle_seconds = self._clock() - self._last_progress
        if idle_seconds > self.stall_seconds:
            if not self._stalled:
                self._report(
                    f"heartbeat paused: main loop idle for {idle_seconds:.0f} s"
                )
            self._stalled = True
            return False
        self._stalled = False
        body = json.dumps({"device_id": self.device_id}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "User-Agent": "edge-ai-heartbeat",
        }
        try:
            self._post(self.url, body, headers, self.timeout_seconds)
        except Exception as exc:
            if not self._failing:
                self._report(f"heartbeat failed: {type(exc).__name__}: {exc}")
            self._failing = True
            return False
        if self._failing:
            self._report("heartbeat restored")
        self._failing = False
        return True

    def _report(self, message: str) -> None:
        if self.reporter is not None:
            self.reporter(message)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.send_once()
            self._stop.wait(self.interval_seconds)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_seconds + 1.0)
