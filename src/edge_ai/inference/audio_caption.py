"""Optional local inference for the MU-NLPC Whisper audio-captioning checkpoint.

The checkpoint is intentionally loaded only when this engine is constructed so
the standard ONNX-only demo and board package do not require PyTorch or
Transformers. It is a laptop evaluation tool: generated text is never a
calibrated safety score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai.inference.base import InferenceEngine, InferenceResult


STYLE_PREFIXES = {
    "clotho": "clotho > caption: ",
    "audiocaps": "audiocaps > caption: ",
    "audioset": "audioset > keywords: ",
}


@dataclass(frozen=True)
class _CaptionRuntime:
    torch: Any
    processor: Any
    tokenizer: Any
    model: Any


RuntimeLoader = Callable[[Path], _CaptionRuntime]


def _load_runtime(model_dir: Path) -> _CaptionRuntime:
    """Load only local checkpoint files; setup downloads happen explicitly."""
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor
    except ImportError as exc:
        raise RuntimeError(
            "audio captioning requires optional dependencies; run "
            "`uv sync --extra audio-caption`"
        ) from exc

    try:
        processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, local_files_only=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"could not load a complete local audio-caption model from {model_dir}; "
            "download the checkpoint before running this command"
        ) from exc
    model.eval()
    return _CaptionRuntime(torch, processor, processor.tokenizer, model)


class WhisperAudioCaptionInferenceEngine(InferenceEngine):
    """Generate one English scene caption from up to 30 seconds of 16 kHz audio."""

    def __init__(
        self,
        model_dir: Path,
        *,
        style: str = "clotho",
        max_new_tokens: int = 64,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"audio-caption model directory not found: {self.model_dir}")
        if style not in STYLE_PREFIXES:
            choices = ", ".join(sorted(STYLE_PREFIXES))
            raise ValueError(f"audio-caption style must be one of: {choices}")
        if isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= 128:
            raise ValueError("audio-caption max_new_tokens must be between 1 and 128")

        self.style = style
        self.max_new_tokens = max_new_tokens
        runtime = (runtime_loader or _load_runtime)(self.model_dir)
        self._torch = runtime.torch
        self._processor = runtime.processor
        self._tokenizer = runtime.tokenizer
        self._model = runtime.model

    def caption(self, data: Any) -> str:
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("audio-caption input must be a non-empty rank-1 waveform")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("audio-caption input waveform must contain only finite values")
        waveform = waveform[: 30 * 16_000]
        features = self._processor(
            waveform, sampling_rate=16_000, return_tensors="pt"
        ).input_features
        with self._torch.inference_mode():
            generated = self._model.generate(
                input_features=features,
                decoder_input_ids=self._decoder_prefix(),
                max_new_tokens=self.max_new_tokens,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != 1:
            raise ValueError("audio-caption model must return one caption")
        caption = decoded[0].strip()
        prefix = STYLE_PREFIXES[self.style].strip()
        if caption.lower().startswith(prefix.lower()):
            caption = caption[len(prefix) :].strip()
        if not caption:
            raise ValueError("audio-caption model produced an empty caption")
        return caption

    def predict(self, data: Any) -> InferenceResult:
        return InferenceResult(
            "caption",
            0.0,
            model_label=self.caption(data),
            source="whisper_audio_caption",
        )

    def _decoder_prefix(self) -> Any:
        prompt_ids = self._processor.get_decoder_prompt_ids(
            language="en", task="transcribe", no_timestamps=True
        )
        initial = [self._model.config.decoder_start_token_id]
        initial.extend(token for _, token in prompt_ids)
        tokenized = self._tokenizer(
            "", text_target=STYLE_PREFIXES[self.style], return_tensors="pt", add_special_tokens=False
        )
        style_tokens = getattr(tokenized, "labels", None)
        if style_tokens is None:
            style_tokens = tokenized.input_ids
        initial_tokens = self._torch.tensor([initial], dtype=self._torch.long)
        return self._torch.cat((initial_tokens, style_tokens), dim=1)
