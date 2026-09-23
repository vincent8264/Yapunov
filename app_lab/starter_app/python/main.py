"""Run the bundled sound pipeline and show matching UNO Q matrix icons.

App Lab runs this file on the board. The bundled ``sound-uno-q.toml`` selects the
input (synthetic sounds, bundled WAV fixtures, or the USB microphone) and the
classifier. Raw samples stay in this process.
"""

from __future__ import annotations

import atexit
import sys
import time
from pathlib import Path


def _bundle() -> tuple[Path, Path]:
    """Return ``(import root, config path)`` for the board bundle or the repo."""
    here = Path(__file__).resolve().parent
    local_config = here / "sound-uno-q.toml"
    if (here / "edge_ai").is_dir() and local_config.is_file():
        return here, local_config
    for parent in here.parents:
        if (parent / "src" / "edge_ai").is_dir() and (
            parent / "configs" / "sound-uno-q.toml"
        ).is_file():
            return parent / "src", parent / "configs" / "sound-uno-q.toml"
    raise RuntimeError(
        "The sound pipeline is not in this App Lab app. "
        "Run scripts/package_app_lab.py and import the zip it writes."
    )


_IMPORT_ROOT, _CONFIG_PATH = _bundle()
sys.path.insert(0, str(_IMPORT_ROOT))

from arduino.app_utils import App  # noqa: E402

from edge_ai.config import load_config  # noqa: E402

_CONFIGURED = load_config(_CONFIG_PATH)
_READY = False


def _cleanup() -> None:
    hardware = _CONFIGURED.pipeline.hardware
    try:
        hardware.shutdown()
    finally:
        close = getattr(_CONFIGURED.pipeline.input_source, "close", None)
        if callable(close):
            close()


def loop() -> None:
    global _READY
    if not _READY:
        _CONFIGURED.pipeline.hardware.check_connection()
        _READY = True
    started = time.perf_counter()
    period = max(
        _CONFIGURED.interval_seconds,
        _CONFIGURED.pipeline.poll_interval_seconds,
    )
    step_result = _CONFIGURED.pipeline.step()
    # Streaming ASR pipelines return once per microphone frame; their YAMNet window
    # is maintained inside the inference adapter while the spectrum keeps updating.
    if step_result is not None:
        result, decision = step_result
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detector_timings = "".join(
            f" {name}_ms={latency_ms:.1f}" for name, latency_ms in result.timings_ms
        )
        transcript = f" transcript={result.transcript}" if result.transcript else ""
        print(
            f"label={result.label} confidence={result.confidence:.3f} "
            f"source={result.source or '-'} model_label={result.model_label or '-'} "
            f"model_confidence="
            f"{result.model_confidence if result.model_confidence is not None else 0.0:.3f} "
            f"action={decision.action} event={decision.event or '-'} "
            f"latency_ms={elapsed_ms:.1f}{detector_timings}{transcript}"
        )
    delay = period - (time.perf_counter() - started)
    if delay > 0.0:
        time.sleep(delay)


atexit.register(_cleanup)
App.run(user_loop=loop)
