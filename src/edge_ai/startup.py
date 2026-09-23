"""Startup checks shared by the board deployment entry point.

The inference pipeline is deliberately usable without a network connection once its
dependencies and model are installed.  The network result is therefore reported to
the operator but is not allowed to block a locally usable safety monitor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from edge_ai.hardware.base import HardwareBackend
from edge_ai.inputs.base import InputSource


INTERNET_CHECK_URL = "https://pypi.org/"
INTERNET_CHECK_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class StartupReport:
    """The completed local and advisory connectivity checks."""

    internet_available: bool
    internet_detail: str


def check_internet(
    *,
    opener: Callable[..., Any] = urlopen,
    url: str = INTERNET_CHECK_URL,
    timeout_seconds: float = INTERNET_CHECK_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Make one small HTTPS request and return its reachability result.

    A response of any HTTP status proves that DNS, TLS, and outbound internet access
    worked.  This check is advisory so an installed, fully local deployment can keep
    monitoring while Wi-Fi is temporarily unavailable.
    """
    # Use GET rather than HEAD because some captive portals and package mirrors
    # reject HEAD even though an ordinary HTTPS request would work.  The response is
    # closed without reading its body.
    request = Request(url, headers={"User-Agent": "edge-ai-startup-check"})
    try:
        response = opener(request, timeout=timeout_seconds)
    except (OSError, URLError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        status = getattr(response, "status", "response")
        return True, f"HTTPS reachable ({status})"
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def run_startup_validation(
    hardware: HardwareBackend,
    input_source: InputSource,
    *,
    on_hardware_ready: Callable[[], None] | None = None,
    internet_checker: Callable[[], tuple[bool, str]] = check_internet,
) -> StartupReport:
    """Verify the board bridge, receive real audio, and report internet status.

    Bridge and microphone failures raise immediately because the monitor cannot
    operate without them.  The model was already opened and validated while the
    configured pipeline was constructed.  Internet reachability is advisory; no
    inference or local alert depends on it after the app has been installed.
    """
    hardware.check_connection()
    if on_hardware_ready is not None:
        on_hardware_ready()
    try:
        input_source.read()
    except Exception as exc:
        raise RuntimeError(f"microphone startup check failed: {exc}") from exc
    internet_available, internet_detail = internet_checker()
    return StartupReport(internet_available, internet_detail)
