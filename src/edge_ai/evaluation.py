"""Hardware-free evaluation helpers for audio classifiers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge_ai.inference.base import InferenceEngine
from edge_ai.inputs.audio import read_wav
from edge_ai.preprocessing.audio import resample_audio_frame


@dataclass(frozen=True)
class KeywordEvaluation:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def precision(self) -> float:
        predicted_positive = self.true_positive + self.false_positive
        return self.true_positive / predicted_positive if predicted_positive else 0.0

    @property
    def recall(self) -> float:
        actual_positive = self.true_positive + self.false_negative
        return self.true_positive / actual_positive if actual_positive else 0.0

    @property
    def accuracy(self) -> float:
        return (
            (self.true_positive + self.true_negative) / self.total
            if self.total
            else 0.0
        )


def evaluate_keyword_dataset(
    dataset: Path,
    engine: InferenceEngine,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 1.0,
    step_seconds: float = 0.2,
) -> KeywordEvaluation:
    """Evaluate WAV clips with overlapping windows, as the live pipeline does."""
    if duration_seconds <= 0.0 or step_seconds <= 0.0:
        raise ValueError("keyword evaluation window and step must be positive")
    root = Path(dataset)
    examples: list[tuple[bool, Path]] = []
    for label, expected_positive in (("help_call", True), ("background", False)):
        directory = root / label
        if not directory.is_dir():
            raise ValueError(f"keyword dataset folder not found: {directory}")
        paths = sorted(directory.rglob("*.wav"))
        if not paths:
            raise ValueError(f"keyword dataset contains no WAV files: {directory}")
        examples.extend((expected_positive, path) for path in paths)

    true_positive = false_positive = true_negative = false_negative = 0
    for expected_positive, path in examples:
        frame = read_wav(path)
        samples = resample_audio_frame(frame, sample_rate=sample_rate)
        window_size = round(sample_rate * duration_seconds)
        step_size = round(sample_rate * step_seconds)
        if window_size < 1 or step_size < 1:
            raise ValueError("keyword evaluation window and step must span a sample")
        starts = list(range(0, max(1, samples.size - window_size + 1), step_size))
        final_start = max(0, samples.size - window_size)
        if starts[-1] != final_start:
            starts.append(final_start)
        predicted_positive = False
        for start in starts:
            waveform = samples[start : start + window_size]
            if waveform.size < window_size:
                waveform = np.pad(waveform, (0, window_size - waveform.size))
            if engine.predict(waveform).label == "help_call":
                predicted_positive = True
                break
        if expected_positive and predicted_positive:
            true_positive += 1
        elif expected_positive:
            false_negative += 1
        elif predicted_positive:
            false_positive += 1
        else:
            true_negative += 1

    return KeywordEvaluation(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )
