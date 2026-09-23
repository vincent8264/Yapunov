from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
        class_labels=tuple(f"class_{index}" for index in range(521)),
        session_factory=factory,
    )
    return engine, session


@pytest.mark.parametrize(
    ("class_index", "expected_label"),
    [
        (393, "smoke_alarm"),
        (437, "glass_break"),
        (454, "fall_thud"),
        (459, "fall_thud"),
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

    assert result.label == expected_label
    assert result.confidence == pytest.approx(0.9)
    assert result.model_label == f"class_{class_index}"
    assert result.model_confidence == pytest.approx(0.9)
    assert result.source == "yamnet"
    assert set(result.evaluated_events or ()) == {
        "smoke_alarm",
        "glass_break",
        "fall_thud",
    }
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
    assert result.model_label == "class_393"
    assert result.model_confidence == pytest.approx(0.04)


_REAL_MODEL = Path(__file__).resolve().parents[1] / "models" / "yamnet.onnx"


@pytest.mark.skipif(not _REAL_MODEL.is_file(), reason="models/yamnet.onnx is not installed")
@pytest.mark.parametrize("event", ["background", "smoke_alarm", "glass_break", "fall_thud"])
def test_real_yamnet_recognizes_most_simulated_sounds(event: str) -> None:
    pytest.importorskip("onnxruntime")
    from edge_ai.inputs.audio import SimulatedSoundInput

    engine = YAMNetInferenceEngine(_REAL_MODEL, background_threshold=0.1)
    source = SimulatedSoundInput([event], seed=1)

    labels = [engine.predict(source.read().samples).label for _ in range(20)]

    assert labels.count(event) >= 16


def test_yamnet_reports_watched_labels_separately_from_events(tmp_path: Path) -> None:
    path = tmp_path / "yamnet.onnx"
    path.write_bytes(b"fake model placeholder")
    scores = np.zeros((2, 521), dtype=np.float32)
    scores[0, 442] = 0.3
    scores[1, 442] = 0.6
    scores[1, 393] = 0.9
    engine = YAMNetInferenceEngine(
        path,
        class_labels=tuple(
            "Drip" if index == 442 else f"class_{index}" for index in range(521)
        ),
        watched_labels=("Drip", "class_450"),
        session_factory=lambda *args, **kwargs: FakeSession(scores),
    )

    result = engine.predict(np.zeros(16_000, dtype=np.float32))

    assert result.label == "smoke_alarm"
    assert dict(result.label_scores) == {
        "Drip": pytest.approx(0.6),
        "class_450": pytest.approx(0.0),
    }


def test_yamnet_rejects_unknown_watched_label(tmp_path: Path) -> None:
    path = tmp_path / "yamnet.onnx"
    path.write_bytes(b"fake model placeholder")

    with pytest.raises(ValueError, match="not in the YAMNet class map: 'Kettle'"):
        YAMNetInferenceEngine(
            path,
            class_labels=tuple(f"class_{index}" for index in range(521)),
            watched_labels=("Kettle",),
            session_factory=lambda *args, **kwargs: FakeSession(
                np.zeros((1, 521), dtype=np.float32)
            ),
        )


def test_real_class_map_contains_configured_risk_labels() -> None:
    from edge_ai.inference.yamnet import load_yamnet_class_labels

    labels = set(load_yamnet_class_labels(_REAL_MODEL.with_name("yamnet_class_map.csv")))

    assert {
        "Drip",
        "Trickle, dribble",
        "Gush",
        "Water tap, faucet",
        "Boiling",
        "Frying (food)",
        "Steam whistle",
    } <= labels


def test_yamnet_rejects_an_invalid_scores_shape(tmp_path: Path) -> None:
    engine, _ = build_engine(tmp_path, np.zeros((1, 520), dtype=np.float32))

    with pytest.raises(ValueError, match=r"\[frames, 521\]"):
        engine.predict(np.zeros(16_000, dtype=np.float32))


def test_yamnet_returns_ranked_raw_classes(tmp_path: Path) -> None:
    scores = np.zeros((2, 521), dtype=np.float32)
    scores[0, 42] = 0.6
    scores[1, 42] = 0.4
    scores[1, 7] = 0.8
    engine, _ = build_engine(tmp_path, scores)

    classes = engine.top_classes(np.zeros(16_000, dtype=np.float32), limit=2)

    assert classes == (
        ("class_7", pytest.approx(0.8)),
        ("class_42", pytest.approx(0.6)),
    )


def test_yamnet_rejects_invalid_top_class_limit(tmp_path: Path) -> None:
    engine, _ = build_engine(tmp_path, np.zeros((1, 521), dtype=np.float32))

    with pytest.raises(ValueError, match="between 1 and 521"):
        engine.top_classes(np.zeros(16_000, dtype=np.float32), limit=0)
