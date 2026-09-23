from pathlib import Path

import numpy as np
import pytest

from edge_ai import evaluation
from edge_ai.inference.base import InferenceResult
from edge_ai.inputs.audio import AudioFrame


class SequenceEngine:
    def __init__(self, labels: list[str]) -> None:
        self._labels = iter(labels)

    def predict(self, data: object) -> InferenceResult:
        label = next(self._labels)
        return InferenceResult(label, 0.9)


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for label in ("help_call", "background"):
        directory = root / label
        directory.mkdir(parents=True)
        (directory / "one.wav").touch()
        (directory / "two.wav").touch()
    return root


def test_evaluate_keyword_dataset_reports_confusion_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluation,
        "read_wav",
        lambda path: AudioFrame(np.zeros(8_000, dtype=np.float32), 8_000),
    )

    result = evaluation.evaluate_keyword_dataset(
        _dataset(tmp_path),
        SequenceEngine(["help_call", "background", "help_call", "background"]),
    )

    assert (result.true_positive, result.false_negative) == (1, 1)
    assert (result.false_positive, result.true_negative) == (1, 1)
    assert result.total == 4
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.accuracy == 0.5


def test_evaluate_keyword_dataset_requires_both_classes(tmp_path: Path) -> None:
    (tmp_path / "help_call").mkdir()
    (tmp_path / "help_call" / "one.wav").touch()

    with pytest.raises(ValueError, match="background"):
        evaluation.evaluate_keyword_dataset(tmp_path, SequenceEngine([]))


def test_evaluate_keyword_dataset_checks_overlapping_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    for label in ("help_call", "background"):
        (root / label).mkdir(parents=True)
        (root / label / "long.wav").touch()
    monkeypatch.setattr(
        evaluation,
        "read_wav",
        lambda path: AudioFrame(np.zeros(32_000, dtype=np.float32), 16_000),
    )
    # The positive is found in its third window; all six negative windows remain
    # background. A first-second-only evaluator would miss this keyword.
    engine = SequenceEngine(
        ["background", "background", "help_call"] + ["background"] * 6
    )

    result = evaluation.evaluate_keyword_dataset(root, engine)

    assert result.true_positive == 1
    assert result.true_negative == 1
