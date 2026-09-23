"""Sequential scheduling for multiple detectors sharing one audio waveform."""

import time
from typing import Any

from edge_ai.inference.base import InferenceEngine, InferenceResult


class ScheduledAudioInferenceEngine(InferenceEngine):
    """Run keyword inference every step and environmental inference periodically."""

    def __init__(
        self,
        environment: InferenceEngine,
        keyword: InferenceEngine,
        *,
        environment_every_steps: int = 3,
    ) -> None:
        if (
            isinstance(environment_every_steps, bool)
            or not isinstance(environment_every_steps, int)
            or environment_every_steps < 1
        ):
            raise ValueError("environment_every_steps must be a positive integer")
        self.environment = environment
        self.keyword = keyword
        self.environment_every_steps = environment_every_steps
        self._step = 0

    def predict(self, data: Any) -> InferenceResult:
        keyword_started = time.perf_counter()
        keyword_result = self.keyword.predict(data)
        timings = [("keyword", (time.perf_counter() - keyword_started) * 1000.0)]
        results = [keyword_result]
        if self._step % self.environment_every_steps == 0:
            environment_started = time.perf_counter()
            results.append(self.environment.predict(data))
            timings.append(
                ("environment", (time.perf_counter() - environment_started) * 1000.0)
            )
        self._step += 1

        evaluated = tuple(
            dict.fromkeys(
                event
                for result in results
                for event in (result.evaluated_events or ())
            )
        )
        chosen = next(
            (result for result in results if result.label == "help_call"),
            next(
                (result for result in results if result.label != "background"),
                keyword_result,
            ),
        )
        sources = "+".join(
            result.source for result in results if result.source is not None
        )
        return InferenceResult(
            chosen.label,
            chosen.confidence,
            model_label=chosen.model_label,
            model_confidence=chosen.model_confidence,
            source=sources or chosen.source,
            evaluated_events=evaluated or chosen.evaluated_events,
            timings_ms=tuple(timings),
            transcript=chosen.transcript,
        )
