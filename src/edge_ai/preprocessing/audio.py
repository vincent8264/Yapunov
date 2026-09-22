"""Model-independent preparation and compact demo features for audio frames."""

from dataclasses import dataclass

import numpy as np

from edge_ai.inputs.audio import AudioFrame


@dataclass(frozen=True)
class AudioFeatures:
    rms: float
    peak: float
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float
    spectral_flatness: float
    zero_crossing_rate: float


def prepare_audio_waveform(
    frame: AudioFrame,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 1.0,
    peak_normalize: bool = False,
) -> np.ndarray:
    """Resample, pad, or trim a frame into a fixed mono float32 waveform."""
    if not isinstance(frame, AudioFrame):
        raise TypeError("audio preprocessing requires an AudioFrame")
    if isinstance(sample_rate, bool) or sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")

    samples = frame.samples
    if frame.sample_rate != sample_rate:
        output_length = max(1, round(samples.size * sample_rate / frame.sample_rate))
        old_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        new_positions = np.linspace(0.0, 1.0, output_length, endpoint=False)
        samples = np.interp(new_positions, old_positions, samples).astype(np.float32)

    required = round(sample_rate * duration_seconds)
    if samples.size < required:
        samples = np.pad(samples, (0, required - samples.size))
    else:
        samples = samples[:required].copy()

    if peak_normalize:
        peak = float(np.max(np.abs(samples)))
        if peak > 0.0:
            samples /= peak
    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)


def extract_audio_features(
    frame: AudioFrame,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 1.0,
) -> AudioFeatures:
    """Extract small spectral features for the model-free integration demo."""
    samples = prepare_audio_waveform(
        frame,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    peak = float(np.max(np.abs(samples)))
    windowed = samples * np.hanning(samples.size)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    total = max(float(power.sum()), np.finfo(np.float64).tiny)

    low = float(power[frequencies < 300.0].sum()) / total
    mid = float(power[(frequencies >= 300.0) & (frequencies < 2_000.0)].sum()) / total
    high = max(0.0, 1.0 - low - mid)
    positive_power = power[1:] + np.finfo(np.float64).tiny
    flatness = float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))
    zero_crossings = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1])))
    return AudioFeatures(rms, peak, low, mid, high, flatness, zero_crossings)
