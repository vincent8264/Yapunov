from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edge_ai.inference.audio_caption import (
    STYLE_PREFIXES,
    WhisperAudioCaptionInferenceEngine,
    _CaptionRuntime,
)


class FakeTorch:
    long = np.int64

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()

    @staticmethod
    def tensor(value: object, *, dtype: object) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def cat(values: tuple[np.ndarray, ...], *, dim: int) -> np.ndarray:
        return np.concatenate(values, axis=dim)


class FakeFeatureExtractor:
    def __call__(self, waveform: np.ndarray, **_: object) -> SimpleNamespace:
        return SimpleNamespace(input_features=waveform)

    @staticmethod
    def get_decoder_prompt_ids(**_: object) -> list[tuple[int, int]]:
        return [(1, 10), (2, 11), (3, 12)]


class FakeTokenizer:
    def __init__(self) -> None:
        self.style_prefix = ""

    def __call__(self, _: str, *, text_target: str, **__: object) -> SimpleNamespace:
        self.style_prefix = text_target
        return SimpleNamespace(labels=np.asarray([[20, 21]], dtype=np.int64))

    def batch_decode(self, _: object, **__: object) -> list[str]:
        return [f"{self.style_prefix}A dog barks near a door."]


class FakeModel:
    config = SimpleNamespace(decoder_start_token_id=9)

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[int]:
        self.calls.append(kwargs)
        return [1]


def _runtime(_: Path) -> _CaptionRuntime:
    return _CaptionRuntime(FakeTorch(), FakeFeatureExtractor(), FakeTokenizer(), FakeModel())


def test_caption_generates_and_strips_requested_style_prefix(tmp_path: Path) -> None:
    engine = WhisperAudioCaptionInferenceEngine(tmp_path, runtime_loader=_runtime)

    result = engine.predict(np.linspace(-1.0, 1.0, 32, dtype=np.float32))

    assert result.label == "caption"
    assert result.confidence == 0.0
    assert result.model_label == "A dog barks near a door."
    assert result.source == "whisper_audio_caption"


def test_caption_uses_the_selected_style_and_limits_audio_to_thirty_seconds(tmp_path: Path) -> None:
    model = FakeModel()

    def runtime(_: Path) -> _CaptionRuntime:
        return _CaptionRuntime(FakeTorch(), FakeFeatureExtractor(), FakeTokenizer(), model)

    engine = WhisperAudioCaptionInferenceEngine(
        tmp_path, style="audioset", runtime_loader=runtime
    )
    engine.caption(np.ones(500_000, dtype=np.float32))

    assert model.calls[0]["input_features"].shape == (480_000,)
    assert STYLE_PREFIXES["audioset"].startswith("audioset")


@pytest.mark.parametrize("style", ("", "unknown"))
def test_caption_rejects_unknown_style(tmp_path: Path, style: str) -> None:
    with pytest.raises(ValueError, match="style"):
        WhisperAudioCaptionInferenceEngine(tmp_path, style=style, runtime_loader=_runtime)


def test_caption_rejects_invalid_waveform(tmp_path: Path) -> None:
    engine = WhisperAudioCaptionInferenceEngine(tmp_path, runtime_loader=_runtime)

    with pytest.raises(ValueError, match="rank-1"):
        engine.caption(np.ones((1, 2), dtype=np.float32))
