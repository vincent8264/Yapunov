from typing import Any

from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.scheduled import ScheduledAudioInferenceEngine


class SequenceEngine(InferenceEngine):
    def __init__(self, results: list[InferenceResult]) -> None:
        self.results = results
        self.calls = 0

    def predict(self, data: Any) -> InferenceResult:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def test_scheduler_runs_keyword_each_step_and_yamnet_periodically() -> None:
    keyword = SequenceEngine(
        [
            InferenceResult(
                "background",
                0.9,
                source="keyword_spotter",
                evaluated_events=("help_call",),
            )
        ]
    )
    environment = SequenceEngine(
        [
            InferenceResult(
                "smoke_alarm",
                0.8,
                source="yamnet",
                evaluated_events=("smoke_alarm", "glass_break", "fall_thud"),
            )
        ]
    )
    scheduler = ScheduledAudioInferenceEngine(
        environment,
        keyword,
        environment_every_steps=3,
    )

    results = [scheduler.predict(object()) for _ in range(4)]

    assert keyword.calls == 4
    assert environment.calls == 2
    assert results[0].label == "smoke_alarm"
    assert results[0].evaluated_events == (
        "help_call",
        "smoke_alarm",
        "glass_break",
        "fall_thud",
    )
    assert results[1].label == "background"
    assert tuple(name for name, _ in results[0].timings_ms) == (
        "keyword",
        "environment",
    )
    assert tuple(name for name, _ in results[1].timings_ms) == ("keyword",)


def test_scheduler_prioritizes_help_when_both_detectors_trigger() -> None:
    keyword = SequenceEngine(
        [
            InferenceResult(
                "help_call",
                0.7,
                source="keyword_spotter",
                evaluated_events=("help_call",),
            )
        ]
    )
    environment = SequenceEngine(
        [
            InferenceResult(
                "glass_break",
                0.9,
                source="yamnet",
                evaluated_events=("smoke_alarm", "glass_break", "fall_thud"),
            )
        ]
    )

    result = ScheduledAudioInferenceEngine(environment, keyword).predict(object())

    assert result.label == "help_call"
    assert result.confidence == 0.7
