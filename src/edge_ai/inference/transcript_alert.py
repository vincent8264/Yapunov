"""Laptop-only combined YAMNet and transcript-trigger inference."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inputs.audio import AudioFrame
from edge_ai.preprocessing.audio import resample_audio_frame
from edge_ai.transcription import SherpaZipformerTranscriber, TranscriptUpdate


_HELP_WORD = re.compile(r"(?<![a-z])help(?![a-z])", flags=re.IGNORECASE)


class TranscriptHelpYAMNetInferenceEngine(InferenceEngine):
    """Fan out raw frames once to ASR and independently to rolling YAMNet windows."""

    def __init__(
        self,
        yamnet: InferenceEngine,
        transcriber: SherpaZipformerTranscriber,
        *,
        window_seconds: float = 1.0,
        yamnet_hop_seconds: float = 0.2,
    ) -> None:
        if window_seconds <= 0.0:
            raise ValueError("YAMNet window duration must be positive")
        if not 0.0 < yamnet_hop_seconds <= window_seconds:
            raise ValueError("YAMNet hop must be positive and no longer than its window")
        self.yamnet = yamnet
        self.transcriber = transcriber
        self.window_samples = round(16_000 * window_seconds)
        self.hop_samples = round(16_000 * yamnet_hop_seconds)
        if self.window_samples < 1 or self.hop_samples < 1:
            raise ValueError("YAMNet window and hop must each span at least one sample")
        self._samples = np.empty(0, dtype=np.float32)
        self._samples_since_yamnet = 0
        self._has_yamnet_window = False

    def predict(self, data: Any) -> InferenceResult:
        if not isinstance(data, AudioFrame):
            raise TypeError("transcript help inference requires an AudioFrame")
        asr_started = time.perf_counter()
        updates = self.transcriber.accept(data)
        timings: list[tuple[str, float]] = [
            ("transcript", (time.perf_counter() - asr_started) * 1000.0)
        ]
        transcript = updates[-1].text if updates else None
        help_detected = self._has_help_word(updates)

        waveform = resample_audio_frame(data, sample_rate=16_000)
        self._samples = np.concatenate((self._samples, waveform))[-self.window_samples :]
        self._samples_since_yamnet += waveform.size
        yamnet_result: InferenceResult | None = None
        if self._ready_for_yamnet():
            yamnet_started = time.perf_counter()
            yamnet_result = self.yamnet.predict(self._samples.copy())
            timings.append(("yamnet", (time.perf_counter() - yamnet_started) * 1000.0))

        evaluated = ("help_call",)
        if yamnet_result is not None:
            evaluated = tuple(
                dict.fromkeys(evaluated + (yamnet_result.evaluated_events or ()))
            )
        source = "zipformer" + (
            f"+{yamnet_result.source or 'yamnet'}" if yamnet_result is not None else ""
        )
        if help_detected:
            return InferenceResult(
                "help_call", 1.0, model_label="help", model_confidence=1.0,
                source=source, evaluated_events=evaluated, timings_ms=tuple(timings),
                transcript=transcript,
            )
        if yamnet_result is not None:
            return InferenceResult(
                yamnet_result.label, yamnet_result.confidence,
                model_label=yamnet_result.model_label,
                model_confidence=yamnet_result.model_confidence,
                source=source, evaluated_events=evaluated, timings_ms=tuple(timings),
                transcript=transcript,
            )
        return InferenceResult(
            "background", 1.0, source=source, evaluated_events=evaluated,
            timings_ms=tuple(timings), transcript=transcript,
        )

    def _ready_for_yamnet(self) -> bool:
        if not self._has_yamnet_window:
            if self._samples.size < self.window_samples:
                return False
            self._has_yamnet_window = True
            self._samples_since_yamnet = 0
            return True
        if self._samples_since_yamnet < self.hop_samples:
            return False
        self._samples_since_yamnet %= self.hop_samples
        return True

    def _has_help_word(self, updates: tuple[TranscriptUpdate, ...]) -> bool:
        return any(_HELP_WORD.search(update.text) for update in updates)
