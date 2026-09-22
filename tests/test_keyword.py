from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edge_ai.inference.keyword import KeywordSpotterInferenceEngine


class Node:
    def __init__(self, name: str, shape: list[int | str]) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, output: np.ndarray, *, input_rank: int = 2) -> None:
        self.output = output
        self.input_rank = input_rank
        self.received: np.ndarray | None = None

    def get_inputs(self) -> list[Node]:
        shape: list[int | str] = ["samples"]
        if self.input_rank == 2:
            shape = [1, "samples"]
        return [Node("waveform", shape)]

    def get_outputs(self) -> list[Node]:
        return [Node("probabilities", [1, 2])]

    def run(
        self, output_names: list[str], feeds: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["probabilities"]
        self.received = feeds["waveform"]
        return [self.output]


def build_engine(
    tmp_path: Path,
    output: np.ndarray,
) -> tuple[KeywordSpotterInferenceEngine, FakeSession]:
    path = tmp_path / "help-kws.onnx"
    path.write_bytes(b"fake")
    session = FakeSession(output)

    def factory(*args: Any, **kwargs: Any) -> FakeSession:
        assert args[0] == str(path)
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        return session

    return (
        KeywordSpotterInferenceEngine(path, session_factory=factory),
        session,
    )


def test_keyword_spotter_detects_help_probability(tmp_path: Path) -> None:
    engine, session = build_engine(
        tmp_path,
        np.array([[0.1, 0.9]], dtype=np.float32),
    )

    result = engine.predict(np.zeros(16_000, dtype=np.float32))

    assert result.label == "help_call"
    assert result.confidence == pytest.approx(0.9)
    assert result.source == "keyword_spotter"
    assert result.evaluated_events == ("help_call",)
    assert session.received is not None
    assert session.received.shape == (1, 16_000)


def test_keyword_spotter_returns_background_below_threshold(tmp_path: Path) -> None:
    engine, _ = build_engine(
        tmp_path,
        np.array([[0.8, 0.2]], dtype=np.float32),
    )

    result = engine.predict(np.zeros(16_000, dtype=np.float32))

    assert result.label == "background"
    assert result.confidence == pytest.approx(0.8)
    assert result.model_confidence == pytest.approx(0.2)


def test_keyword_spotter_rejects_logits(tmp_path: Path) -> None:
    engine, _ = build_engine(
        tmp_path,
        np.array([[-2.0, 3.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="probability"):
        engine.predict(np.zeros(16_000, dtype=np.float32))


def test_keyword_spotter_rejects_invalid_unselected_score(tmp_path: Path) -> None:
    engine, _ = build_engine(
        tmp_path,
        np.array([[2.0, 0.4]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="probability"):
        engine.predict(np.zeros(16_000, dtype=np.float32))
