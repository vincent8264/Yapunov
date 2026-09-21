"""Lifecycle-aware execution for a configured pipeline."""

from collections.abc import Callable
import time

from edge_ai.config import ConfiguredPipeline


def _cleanup(configured: ConfiguredPipeline) -> None:
    hardware = configured.pipeline.hardware
    input_source = configured.pipeline.input_source
    try:
        hardware.set_led(False)
    finally:
        close = getattr(input_source, "close", None)
        if callable(close):
            close()


def run_pipeline(
    configured: ConfiguredPipeline,
    *,
    max_steps: int | None = None,
    emit: Callable[[str], None] = print,
) -> int:
    """Run until interrupted or ``max_steps`` is reached, then clean up."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    steps = 0
    try:
        while max_steps is None or steps < max_steps:
            started = time.perf_counter()
            result, decision = configured.pipeline.step()
            steps += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            emit(
                f"step={steps} label={result.label} confidence={result.confidence:.3f} "
                f"action={decision.action} latency_ms={elapsed_ms:.1f}"
            )
            delay = configured.interval_seconds - (time.perf_counter() - started)
            if delay > 0.0 and (max_steps is None or steps < max_steps):
                time.sleep(delay)
    finally:
        _cleanup(configured)
    return steps
