import numpy as np

from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.transcript_alert import TranscriptHelpYAMNetInferenceEngine
from edge_ai.inputs.audio import AudioFrame
from edge_ai.sound_decision import SoundDecisionPolicy
from edge_ai.transcription import TranscriptUpdate


class _FakeYAMNet(InferenceEngine):
    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def predict(self, data: object) -> InferenceResult:
        waveform = np.asarray(data)
        self.windows.append(waveform)
        return InferenceResult(
            "smoke_alarm",
            0.9,
            source="yamnet",
            evaluated_events=("smoke_alarm", "glass_break", "fall_thud"),
        )


class _FakeTranscriber:
    def __init__(self, updates: list[tuple[TranscriptUpdate, ...]]) -> None:
        self.updates = iter(updates)
        self.frames: list[AudioFrame] = []

    def accept(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]:
        self.frames.append(frame)
        return next(self.updates, ())


def _frame() -> AudioFrame:
    return AudioFrame(np.zeros(3_200, dtype=np.float32), 16_000)


def test_help_transcript_is_a_sound_decision_event() -> None:
    transcriber = _FakeTranscriber([(TranscriptUpdate("please help me", False),)])
    engine = TranscriptHelpYAMNetInferenceEngine(_FakeYAMNet(), transcriber)  # type: ignore[arg-type]

    result = engine.predict(_frame())
    policy = SoundDecisionPolicy({"help_call": 1.0}, confirmations=1, hold_seconds=0.0)

    assert result.label == "help_call"
    assert result.confidence == 1.0
    assert result.transcript == "please help me"
    assert result.evaluated_events == ("help_call",)
    assert policy(result).event == "help_call"


def test_help_detection_requires_a_standalone_word() -> None:
    transcriber = _FakeTranscriber([(TranscriptUpdate("helping now", False),)])
    engine = TranscriptHelpYAMNetInferenceEngine(_FakeYAMNet(), transcriber)  # type: ignore[arg-type]

    assert engine.predict(_frame()).label == "background"


def test_help_detection_does_not_suppress_repeated_transcript_updates() -> None:
    transcriber = _FakeTranscriber(
        [
            (TranscriptUpdate("how help", False),),
            (TranscriptUpdate("how help help", True),),
        ]
    )
    engine = TranscriptHelpYAMNetInferenceEngine(_FakeYAMNet(), transcriber)  # type: ignore[arg-type]

    assert engine.predict(_frame()).label == "help_call"
    assert engine.predict(_frame()).label == "help_call"


def test_asr_frames_are_not_replayed_and_yamnet_uses_rolling_window() -> None:
    yamnet = _FakeYAMNet()
    transcriber = _FakeTranscriber([()] * 6)
    engine = TranscriptHelpYAMNetInferenceEngine(yamnet, transcriber)  # type: ignore[arg-type]

    results = [engine.predict(_frame()) for _ in range(6)]

    assert len(transcriber.frames) == 6
    assert len(yamnet.windows) == 2
    assert all(window.shape == (16_000,) for window in yamnet.windows)
    assert results[3].label == "background"
    assert results[4].label == "smoke_alarm"
    assert results[5].label == "smoke_alarm"
