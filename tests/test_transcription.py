from pathlib import Path

import numpy as np
import pytest

from edge_ai.inputs.audio import AudioFrame
from edge_ai.transcription import SherpaZipformerTranscriber, ZipformerModelPaths


class _FakeStream:
    def __init__(self) -> None:
        self.accepted: list[tuple[int, np.ndarray]] = []
        self.finished = False

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.accepted.append((sample_rate, samples))

    def input_finished(self) -> None:
        self.finished = True


class _FakeRecognizer:
    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.text = ""
        self.endpoint = False
        self.decoded = 0
        self.resets = 0

    def create_stream(self) -> _FakeStream:
        return self.stream

    def is_ready(self, stream: _FakeStream) -> bool:
        return self.decoded == 0 and bool(stream.accepted)

    def decode_stream(self, stream: _FakeStream) -> None:
        self.decoded += 1

    def get_result(self, stream: _FakeStream) -> str:
        return self.text

    def is_endpoint(self, stream: _FakeStream) -> bool:
        return self.endpoint

    def reset(self, stream: _FakeStream) -> None:
        self.resets += 1


def _model_dir(tmp_path: Path) -> Path:
    for name in (
        "tokens.txt",
        "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    ):
        (tmp_path / name).touch()
    return tmp_path


def test_model_paths_require_all_model_assets(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Sherpa Zipformer model file not found"):
        ZipformerModelPaths.from_directory(tmp_path)


def test_transcriber_emits_changed_partial_and_final_text(tmp_path: Path) -> None:
    recognizer = _FakeRecognizer()
    transcriber = SherpaZipformerTranscriber(
        _model_dir(tmp_path), recognizer_factory=lambda **_: recognizer
    )
    frame = AudioFrame(np.zeros(3, dtype=np.float32), 16_000)

    recognizer.text = "hello"
    assert transcriber.accept(frame)[0].text == "hello"
    assert transcriber.accept(frame) == ()

    recognizer.text = "hello world"
    recognizer.endpoint = True
    updates = transcriber.accept(frame)
    assert updates[0].text == "hello world"
    assert updates[0].is_final is True
    assert recognizer.resets == 1


def test_transcriber_rejects_bad_thread_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="thread count"):
        SherpaZipformerTranscriber(_model_dir(tmp_path), num_threads=0)
