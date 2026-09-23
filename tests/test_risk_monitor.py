import pytest

from edge_ai.decision import Decision
from edge_ai.inference.base import InferenceResult
from edge_ai.risk_monitor import RiskRule, RiskWarningDecision, RollingRiskMonitor
from edge_ai.sound_decision import SoundDecisionPolicy

WATER = RiskRule(
    name="water_leak",
    message="Possible water leak detected",
    labels=("Drip", "Trickle, dribble", "Gush", "Water tap, faucet"),
    threshold=0.2,
    window_seconds=15.0,
    min_detected_seconds=7.0,
    cooldown_seconds=60.0,
)
COOKING = RiskRule(
    name="unattended_cooking",
    message="Possible unattended cooking detected",
    labels=("Boiling", "Frying (food)", "Steam whistle"),
    threshold=0.2,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def feed(
    monitor: RollingRiskMonitor,
    clock: Clock,
    scores: dict[str, float],
    *,
    seconds: float,
    step: float = 0.25,
) -> list[str]:
    triggered: list[str] = []
    for _ in range(round(seconds / step)):
        clock.now += step
        triggered += [warning.risk for warning in monitor.update(scores)]
    return triggered


def test_one_detection_does_not_warn() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER], clock=clock)

    assert feed(monitor, clock, {"Drip": 0.9}, seconds=0.2) == []
    assert feed(monitor, clock, {}, seconds=10.0) == []


def test_sustained_water_sound_warns_once_after_seven_seconds() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER], clock=clock)

    assert feed(monitor, clock, {"Water tap, faucet": 0.5}, seconds=7.0) == []
    assert monitor.detected_seconds("water_leak") == pytest.approx(6.75)
    assert feed(monitor, clock, {"Gush": 0.5}, seconds=0.25) == ["water_leak"]
    assert feed(monitor, clock, {"Drip": 0.5}, seconds=20.0) == []


def test_intermittent_detections_accumulate_within_the_window() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER], clock=clock)
    triggered: list[str] = []

    for _ in range(8):
        triggered += feed(monitor, clock, {"Drip": 0.5}, seconds=1.0)
        triggered += feed(monitor, clock, {"Drip": 0.05}, seconds=0.5)

    assert triggered == ["water_leak"]


def test_old_observations_are_pruned_and_stop_counting() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER], clock=clock)

    feed(monitor, clock, {"Drip": 0.5}, seconds=5.0)
    assert monitor.detected_seconds("water_leak") == pytest.approx(4.75)
    feed(monitor, clock, {}, seconds=15.0)

    assert monitor.detected_seconds("water_leak") == 0.0
    assert all(
        item.timestamp > clock.now - WATER.window_seconds
        for item in monitor.history("water_leak")
    )
    assert feed(monitor, clock, {"Drip": 0.5}, seconds=5.0) == []


def test_cooldown_allows_a_repeat_warning_later() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER], clock=clock)

    triggered = feed(monitor, clock, {"Drip": 0.5}, seconds=70.0)

    assert triggered == ["water_leak", "water_leak"]


def test_stalled_input_gap_is_not_credited_as_sustained_sound() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER], max_observation_seconds=1.0, clock=clock)

    monitor.update({"Drip": 0.5})
    clock.now = 10.0
    monitor.update({"Drip": 0.5})

    assert monitor.detected_seconds("water_leak") == pytest.approx(1.0)


def test_risks_are_tracked_independently() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([WATER, COOKING], clock=clock)

    triggered = feed(
        monitor, clock, {"Frying (food)": 0.6, "Drip": 0.1}, seconds=8.0
    )

    assert triggered == ["unattended_cooking"]
    assert monitor.detected_seconds("water_leak") == 0.0


def test_wrapper_adds_warning_without_changing_emergency_decision() -> None:
    clock = Clock()
    emergency = SoundDecisionPolicy(
        {"smoke_alarm": 0.5}, confirmations=1, hold_seconds=0.0, clock=clock
    )
    decide = RiskWarningDecision(emergency, RollingRiskMonitor([COOKING], clock=clock))
    decisions: list[Decision] = []

    for _ in range(40):
        clock.now += 0.2
        decisions.append(
            decide(InferenceResult("background", 0.9, label_scores=(("Boiling", 0.7),)))
        )
    alarm = decide(
        InferenceResult("smoke_alarm", 0.8, label_scores=(("Boiling", 0.7),))
    )

    warned = [decision for decision in decisions if decision.warnings]
    assert len(warned) == 1
    assert warned[0].action == "idle"
    assert warned[0].notify is False
    (warning,) = warned[0].warnings
    assert warning.risk == "unattended_cooking"
    assert warning.message == "Possible unattended cooking detected"
    assert warning.detected_seconds >= 7.0
    assert warning.window_seconds == 15.0
    assert warning.peak_label == "Boiling"
    assert warning.peak_score == pytest.approx(0.7)
    assert alarm.action == "alert"
    assert alarm.event == "smoke_alarm"
    assert alarm.warnings == ()


def test_wrapper_ignores_results_without_label_scores() -> None:
    clock = Clock()
    monitor = RollingRiskMonitor([COOKING], clock=clock)
    decide = RiskWarningDecision(lambda _result: Decision("idle"), monitor)

    decide(InferenceResult("background", 0.9, source="keyword_spotter"))

    assert monitor.history("unattended_cooking") == ()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"labels": ()}, "at least one model label"),
        ({"threshold": 1.5}, "between 0 and 1"),
        ({"window_seconds": 0.0}, "window_seconds must be positive"),
        ({"min_detected_seconds": 20.0}, "no longer than window_seconds"),
        ({"min_detected_seconds": 0.0}, "min_detected_seconds must be positive"),
        ({"cooldown_seconds": -1.0}, "cooldown_seconds must be non-negative"),
        ({"message": ""}, "message must not be empty"),
    ],
)
def test_invalid_rules_are_rejected(changes: dict[str, object], message: str) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match=message):
        RollingRiskMonitor([replace(WATER, **changes)])


def test_duplicate_risk_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        RollingRiskMonitor([WATER, WATER])
