"""Lifecycle-aware execution for a configured pipeline."""

from collections.abc import Callable
import sys
import time

from edge_ai.config import ConfiguredPipeline


def _cleanup(configured: ConfiguredPipeline) -> None:
    hardware = configured.pipeline.hardware
    input_source = configured.pipeline.input_source
    try:
        hardware.shutdown()
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
        configured.pipeline.hardware.check_connection()
        while max_steps is None or steps < max_steps:
            started = time.perf_counter()
            step_result = configured.pipeline.step()
            if step_result is None:
                delay = max(configured.interval_seconds, configured.pipeline.poll_interval_seconds) - (
                    time.perf_counter() - started
                )
                if delay > 0.0:
                    time.sleep(delay)
                continue
            result, decision = step_result
            steps += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            emit(
                f"step={steps} label={result.label} confidence={result.confidence:.3f} "
                f"yamnet_label={result.model_label or '-'} "
                f"yamnet_confidence={result.model_confidence if result.model_confidence is not None else 0.0:.3f} "
                f"action={decision.action} event={decision.event or '-'} "
                f"notify={str(decision.notify).lower()} latency_ms={elapsed_ms:.1f}"
            )
            delay = max(
                configured.interval_seconds, configured.pipeline.poll_interval_seconds
            ) - (time.perf_counter() - started)
            if delay > 0.0 and (max_steps is None or steps < max_steps):
                time.sleep(delay)
    finally:
        active_exception = sys.exception()
        try:
            _cleanup(configured)
        except Exception as cleanup_error:
            if active_exception is None:
                raise
            active_exception.add_note(
                f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )
    return steps
