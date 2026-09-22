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


def resample_audio(
    samples: np.ndarray,
    *,
    source_rate: int,
    target_rate: int,
    half_width: int = 32,
) -> np.ndarray:
    """Band-limit and resample mono audio with a windowed-sinc filter."""
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("resampling requires a non-empty mono waveform")
    if any(isinstance(rate, bool) or rate < 1 for rate in (source_rate, target_rate)):
        raise ValueError("resampling rates must be positive integers")
    if isinstance(half_width, bool) or half_width < 2:
        raise ValueError("resampling half_width must be at least 2")
    if source_rate == target_rate:
        return waveform.copy()

    output_length = max(1, round(waveform.size * target_rate / source_rate))
    rate_ratio = target_rate / source_rate
    # Stay slightly below the lower Nyquist limit so the finite filter has room
    # to roll off before content would alias during downsampling.
    cutoff = 0.5 * min(1.0, rate_ratio) * 0.94
    offsets = np.arange(-half_width + 1, half_width + 1, dtype=np.int64)
    output = np.empty(output_length, dtype=np.float32)

    # Work in blocks to keep the temporary index/kernel arrays small on the board.
    for start in range(0, output_length, 4096):
        stop = min(start + 4096, output_length)
        positions = np.arange(start, stop, dtype=np.float64) / rate_ratio
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        distances = indices - positions[:, None]
        in_window = np.abs(distances) <= half_width
        window = np.where(
            in_window,
            0.5 + 0.5 * np.cos(np.pi * distances / half_width),
            0.0,
        )
        weights = 2.0 * cutoff * np.sinc(2.0 * cutoff * distances) * window
        valid = (indices >= 0) & (indices < waveform.size)
        weights *= valid
        safe_indices = np.clip(indices, 0, waveform.size - 1)
        weight_sum = weights.sum(axis=1)
        output[start:stop] = (
            (waveform[safe_indices] * weights).sum(axis=1) / weight_sum
        ).astype(np.float32)

    return output


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
        samples = resample_audio(
            samples,
            source_rate=frame.sample_rate,
            target_rate=sample_rate,
        )

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
