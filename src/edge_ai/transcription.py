"""Local, streaming speech-to-text helpers for laptop evaluation.

This module deliberately stays separate from the alert inference contract: a
transcript is user-facing diagnostic data, not an input to the safety decision
policy.  It accepts each microphone frame once, so it can later be fanned out
alongside YAMNet without replaying overlapping YAMNet windows into the ASR
stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai.inputs.audio import AudioFrame


@dataclass(frozen=True)
class TranscriptUpdate:
    """A changed partial transcript or a completed endpoint segment."""

    text: str
    is_final: bool


@dataclass(frozen=True)
class ZipformerModelPaths:
    """The four model assets required by the selected 66M Zipformer release."""

    tokens: Path
    encoder: Path
    decoder: Path
    joiner: Path

    @classmethod
    def from_directory(cls, directory: Path) -> "ZipformerModelPaths":
        root = Path(directory)
        paths = cls(
            tokens=root / "tokens.txt",
            encoder=root / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            decoder=root / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            joiner=root / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        )
        missing = next((path for path in paths.__dict__.values() if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(
                f"Sherpa Zipformer model file not found: {missing}. "
                "Download and unpack sherpa-onnx-streaming-zipformer-en-2023-06-26 "
                "or pass its directory with --model-dir."
            )
        return paths


class SherpaZipformerTranscriber:
    """Incrementally decode 16 kHz microphone audio with a Zipformer transducer."""

    def __init__(
        self,
        model_dir: Path,
        *,
        num_threads: int = 1,
        recognizer_factory: Any | None = None,
    ) -> None:
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("ASR thread count must be a positive integer")
        paths = ZipformerModelPaths.from_directory(model_dir)
        if recognizer_factory is None:
            try:
                _load_project_onnxruntime_on_windows()
                import sherpa_onnx
            except ImportError as exc:
                raise RuntimeError(
                    "transcription requires the optional 'asr' dependency; run "
                    "'uv sync --extra asr' first"
                ) from exc
            recognizer_factory = sherpa_onnx.OnlineRecognizer.from_transducer
        try:
            self._recognizer = recognizer_factory(
                tokens=str(paths.tokens),
                encoder=str(paths.encoder),
                decoder=str(paths.decoder),
                joiner=str(paths.joiner),
                num_threads=num_threads,
                provider="cpu",
                sample_rate=16_000,
                feature_dim=80,
                decoding_method="greedy_search",
            )
            self._stream = self._recognizer.create_stream()
        except Exception as exc:
            raise RuntimeError(f"could not initialize Sherpa Zipformer ASR: {exc}") from exc
        self._last_partial = ""

    def accept(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]:
        """Feed one frame once and return only text which changed since last call."""
        if not isinstance(frame, AudioFrame):
            raise TypeError("Sherpa transcription requires an AudioFrame")
        samples = np.ascontiguousarray(frame.samples, dtype=np.float32)
        self._stream.accept_waveform(frame.sample_rate, samples)
        self._decode_ready()
        text = self._text()
        updates: list[TranscriptUpdate] = []
        if self._recognizer.is_endpoint(self._stream):
            if text:
                updates.append(TranscriptUpdate(text, is_final=True))
            self._recognizer.reset(self._stream)
            self._last_partial = ""
        elif text and text != self._last_partial:
            updates.append(TranscriptUpdate(text, is_final=False))
            self._last_partial = text
        return tuple(updates)

    def finish(self) -> TranscriptUpdate | None:
        """Flush pending speech when the caller stops capture."""
        self._stream.accept_waveform(16_000, np.zeros(10_560, dtype=np.float32))
        self._stream.input_finished()
        self._decode_ready()
        text = self._text()
        self._last_partial = ""
        return TranscriptUpdate(text, is_final=True) if text else None

    def _decode_ready(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _text(self) -> str:
        result = self._recognizer.get_result(self._stream)
        text = getattr(result, "text", result)
        return str(text).strip()


def _load_project_onnxruntime_on_windows() -> None:
    """Preload this environment's ONNX Runtime before Sherpa's extension loads.

    Some Windows installations contain an obsolete ``onnxruntime.dll`` in
    ``System32``.  Windows may otherwise bind Sherpa's extension to that DLL
    instead of this project's declared ONNX Runtime, which produces an opaque
    unsupported-API error while loading modern model files.
    """
    if os.name != "nt":
        return
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("Sherpa transcription requires the 'onnxruntime' package") from exc
    runtime = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
    if not runtime.is_file():
        raise RuntimeError(f"ONNX Runtime DLL not found: {runtime}")
    ctypes.WinDLL(str(runtime))
