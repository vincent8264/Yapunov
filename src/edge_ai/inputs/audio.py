"""Audio inputs for deterministic replay, live microphones, and laptop demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import wave
from typing import Any, Sequence

import numpy as np

from edge_ai.inputs.base import InputSource


@dataclass(frozen=True)
class AudioFrame:
    """One mono audio frame whose samples are normalized to roughly ``[-1, 1]``."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("audio samples must be a non-empty mono array")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio samples must contain only finite values")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("audio sample_rate must be a positive integer")
        object.__setattr__(self, "samples", samples)


def _decode_pcm(payload: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width * 8} bits")


def read_wav(path: Path) -> AudioFrame:
    """Read an uncompressed PCM WAV file and downmix it to mono."""
    wav_path = Path(path)
    try:
        with wave.open(str(wav_path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise ValueError(f"compressed WAV files are not supported: {wav_path}")
            channels = source.getnchannels()
            if channels < 1:
                raise ValueError(f"WAV file has no channels: {wav_path}")
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            payload = source.readframes(source.getnframes())
    except (FileNotFoundError, wave.Error) as exc:
        raise ValueError(f"could not read WAV file {wav_path}: {exc}") from exc

    samples = _decode_pcm(payload, sample_width)
    if samples.size % channels:
        raise ValueError(f"WAV sample data is incomplete: {wav_path}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return AudioFrame(samples=samples, sample_rate=sample_rate)


class WavAudioInput(InputSource):
    """Replay WAV files in a stable order for demos and regression tests."""

    def __init__(self, paths: Sequence[Path], *, loop: bool = True) -> None:
        if not paths:
            raise ValueError("WAV input requires at least one path")
        self.paths = tuple(Path(path) for path in paths)
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise ValueError(f"WAV file not found: {missing[0]}")
        self.loop = loop
        self._index = 0

    def read(self) -> AudioFrame:
        if self._index >= len(self.paths):
            if not self.loop:
                raise RuntimeError("WAV input is exhausted")
            self._index = 0
        path = self.paths[self._index]
        self._index += 1
        return read_wav(path)


class MicrophoneInput(InputSource):
    """Continuously capture fixed-size frames from a local USB microphone."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        duration_seconds: float = 1.0,
        device: str | int | None = None,
        backend: Any | None = None,
    ) -> None:
        if isinstance(sample_rate, bool) or sample_rate < 1:
            raise ValueError("microphone sample_rate must be positive")
        if duration_seconds <= 0.0:
            raise ValueError("microphone duration_seconds must be positive")
        if backend is None:
            try:
                import sounddevice as backend
            except ImportError as exc:
                raise RuntimeError(
                    "microphone input requires the 'sounddevice' package"
                ) from exc
        self.sample_rate = sample_rate
        self.frame_count = round(sample_rate * duration_seconds)
        try:
            self._stream = backend.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_count,
                device=device,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"could not open microphone {device!r}: {exc}") from exc

    def read(self) -> AudioFrame:
        samples, overflowed = self._stream.read(self.frame_count)
        if overflowed:
            raise RuntimeError("microphone input overflowed; audio samples were lost")
        return AudioFrame(np.asarray(samples, dtype=np.float32).reshape(-1), self.sample_rate)

    def close(self) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()


class SimulatedSoundInput(InputSource):
    """Generate repeatable acoustic signatures to exercise the whole pipeline."""

    SUPPORTED_EVENTS = frozenset({"background", "smoke_alarm", "glass_break", "fall_thud"})

    def __init__(
        self,
        events: Sequence[str],
        *,
        sample_rate: int = 16_000,
        duration_seconds: float = 1.0,
        seed: int = 7,
        loop: bool = True,
    ) -> None:
        if not events:
            raise ValueError("simulated sound input requires at least one event")
        unsupported = [event for event in events if event not in self.SUPPORTED_EVENTS]
        if unsupported:
            raise ValueError(f"unsupported simulated sound event: {unsupported[0]!r}")
        if isinstance(sample_rate, bool) or sample_rate < 8_000:
            raise ValueError("simulated sound sample_rate must be at least 8000")
        if duration_seconds <= 0.0:
            raise ValueError("simulated sound duration_seconds must be positive")
        self.events = tuple(events)
        self.sample_rate = sample_rate
        self.sample_count = round(sample_rate * duration_seconds)
        self.loop = loop
        self._index = 0
        self._random = random.Random(seed)

    def read(self) -> AudioFrame:
        if self._index >= len(self.events):
            if not self.loop:
                raise RuntimeError("simulated sound input is exhausted")
            self._index = 0
        event = self.events[self._index]
        self._index += 1
        seed = self._random.randrange(0, 2**32)
        generator = np.random.default_rng(seed)
        time_axis = np.arange(self.sample_count, dtype=np.float32) / self.sample_rate

        if event == "background":
            samples = generator.normal(0.0, 0.003, self.sample_count)
        elif event == "smoke_alarm":
            carrier = np.sin(2.0 * np.pi * 3_000.0 * time_axis)
            pulse = (np.sin(2.0 * np.pi * 2.0 * time_axis) > -0.2).astype(np.float32)
            samples = 0.75 * carrier * pulse
        elif event == "glass_break":
            noise = generator.normal(0.0, 1.0, self.sample_count)
            spectrum = np.fft.rfft(noise)
            frequencies = np.fft.rfftfreq(self.sample_count, 1.0 / self.sample_rate)
            spectrum[frequencies < 2_000.0] = 0.0
            samples = np.fft.irfft(spectrum, self.sample_count) * np.exp(-7.0 * time_axis)
        else:
            carrier = np.sin(2.0 * np.pi * 85.0 * time_axis)
            samples = 0.9 * carrier * np.exp(-5.0 * time_axis)

        peak = max(float(np.max(np.abs(samples))), 1.0)
        return AudioFrame((samples / peak).astype(np.float32), self.sample_rate)
