from edge_ai.decision import Decision
from edge_ai.inference.base import InferenceResult
from edge_ai.sound_decision import SoundDecisionPolicy


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_policy_requires_confirmation_and_marks_only_new_notification() -> None:
    clock = Clock()
    policy = SoundDecisionPolicy(
        {"glass_break": 0.8},
        confirmations=2,
        hold_seconds=0.0,
        notification_cooldown_seconds=10.0,
        clock=clock,
    )

    assert policy(InferenceResult("glass_break", 0.9)) == Decision("idle")
    first = policy(InferenceResult("glass_break", 0.9))
    repeated = policy(InferenceResult("glass_break", 0.9))

    assert first == Decision("alert", event="glass_break", confidence=0.9, notify=True)
    assert repeated == Decision("alert", event="glass_break", confidence=0.9, notify=False)

    clock.now = 10.0
    assert policy(InferenceResult("glass_break", 0.9)).notify is True


def test_policy_holds_visual_alert_after_sound_ends() -> None:
    clock = Clock()
    policy = SoundDecisionPolicy(
        {"fall_thud": 0.8},
        confirmations=1,
        hold_seconds=3.0,
        clock=clock,
    )

    policy(InferenceResult("fall_thud", 0.95))
    clock.now = 2.0
    held = policy(InferenceResult("background", 0.99))
    clock.now = 3.0
    cleared = policy(InferenceResult("background", 0.99))

    assert held.action == "alert"
    assert held.event == "fall_thud"
    assert held.notify is False
    assert cleared == Decision("idle")


def test_policy_supports_per_event_confirmation_counts() -> None:
    policy = SoundDecisionPolicy(
        {"smoke_alarm": 0.8, "glass_break": 0.8},
        confirmations={"smoke_alarm": 2, "glass_break": 1},
        hold_seconds=0.0,
    )

    assert policy(InferenceResult("smoke_alarm", 0.9)) == Decision("idle")
    glass = policy(InferenceResult("glass_break", 0.9))

    assert glass == Decision("alert", event="glass_break", confidence=0.9, notify=True)


def test_policy_requires_confirmation_keys_to_match_thresholds() -> None:
    try:
        SoundDecisionPolicy(
            {"smoke_alarm": 0.8, "glass_break": 0.8},
            confirmations={"smoke_alarm": 2},
        )
    except ValueError as exc:
        assert "match threshold events" in str(exc)
    else:
        raise AssertionError("incomplete confirmation mapping was accepted")


def test_detector_specific_background_only_resets_its_own_candidates() -> None:
    policy = SoundDecisionPolicy(
        {"smoke_alarm": 0.8, "help_call": 0.5},
        confirmations={"smoke_alarm": 2, "help_call": 2},
        hold_seconds=0.0,
    )

    first_alarm = policy(
        InferenceResult(
            "smoke_alarm",
            0.9,
            source="yamnet",
            evaluated_events=("smoke_alarm",),
        )
    )
    keyword_negative = policy(
        InferenceResult(
            "background",
            0.9,
            source="keyword_spotter",
            evaluated_events=("help_call",),
        )
    )
    confirmed_alarm = policy(
        InferenceResult(
            "smoke_alarm",
            0.9,
            source="yamnet",
            evaluated_events=("smoke_alarm",),
        )
    )

    assert first_alarm.action == "idle"
    assert keyword_negative.action == "idle"
    assert confirmed_alarm.event == "smoke_alarm"


def test_policy_rejects_unsafe_threshold() -> None:
    try:
        SoundDecisionPolicy({"smoke_alarm": 1.1})
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("invalid threshold was accepted")
