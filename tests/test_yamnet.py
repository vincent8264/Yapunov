from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edge_ai.inference.base import InferenceResult
from edge_ai.inference.yamnet import YAMNetInferenceEngine


class Node:
    def __init__(self, name: str, shape: list[int | str]) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores
        self.feeds: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[Node]:
        return [Node("waveform", ["samples"])]

    def get_outputs(self) -> list[Node]:
        return [
            Node("output_0", ["frames", 521]),
            Node("output_1", ["frames", 1024]),
            Node("output_2", ["spectrogram_frames", 64]),
        ]

    def run(
        self,
        output_names: list[str],
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names == ["output_0"]
        self.feeds = feeds
        return [self.scores]


def build_engine(
    tmp_path: Path,
    scores: np.ndarray,
    *,
    background_threshold: float = 0.1,
) -> tuple[YAMNetInferenceEngine, FakeSession]:
    path = tmp_path / "yamnet.onnx"
    path.write_bytes(b"fake model placeholder")
    session = FakeSession(scores)

    def factory(*args: Any, **kwargs: Any) -> FakeSession:
        assert args[0] == str(path)
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        return session

    engine = YAMNetInferenceEngine(
        path,
        background_threshold=background_threshold,
        session_factory=factory,
    )
    return engine, session


@pytest.mark.parametrize(
    ("class_index", "expected_label"),
    [
        (393, "smoke_alarm"),
        (437, "glass_break"),
        (454, "fall_thud"),
    ],
)
def test_yamnet_maps_audioset_classes_to_project_events(
    tmp_path: Path,
    class_index: int,
    expected_label: str,
) -> None:
    scores = np.zeros((2, 521), dtype=np.float32)
    scores[1, class_index] = 0.9
    engine, session = build_engine(tmp_path, scores)

    result = engine.predict(np.zeros(16_000, dtype=np.float32))

    assert result == InferenceResult(expected_label, pytest.approx(0.9))
    assert session.feeds is not None
    assert session.feeds["waveform"].shape == (16_000,)
    assert session.feeds["waveform"].dtype == np.float32


def test_yamnet_returns_background_below_target_floor(tmp_path: Path) -> None:
    scores = np.zeros((1, 521), dtype=np.float32)
    scores[0, 393] = 0.04
    engine, _ = build_engine(tmp_path, scores, background_threshold=0.1)

    result = engine.predict(np.zeros(16_000, dtype=np.float32))

    assert result.label == "background"
    assert result.confidence == pytest.approx(0.96)


def test_yamnet_rejects_an_invalid_scores_shape(tmp_path: Path) -> None:
    engine, _ = build_engine(tmp_path, np.zeros((1, 520), dtype=np.float32))

    with pytest.raises(ValueError, match=r"\[frames, 521\]"):
        engine.predict(np.zeros(16_000, dtype=np.float32))
