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


def resample_audio_frame(frame: AudioFrame, *, sample_rate: int = 16_000) -> np.ndarray:
    """Return the complete mono frame at ``sample_rate`` without padding or trimming."""
    if not isinstance(frame, AudioFrame):
        raise TypeError("audio preprocessing requires an AudioFrame")
    if isinstance(sample_rate, bool) or sample_rate < 1:
        raise ValueError("sample_rate must be positive")

    return resample_audio(
        frame.samples,
        source_rate=frame.sample_rate,
        target_rate=sample_rate,
    )


class SlidingAudioWindow:
    """Build an overlapping fixed-size waveform from smaller captured audio chunks."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        window_seconds: float = 1.0,
        peak_normalize: bool = False,
    ) -> None:
        if isinstance(sample_rate, bool) or sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        self.sample_rate = sample_rate
        self.window_samples = round(sample_rate * window_seconds)
        self.peak_normalize = peak_normalize
        self._samples = np.empty(0, dtype=np.float32)

    def __call__(self, frame: AudioFrame) -> np.ndarray:
        chunk = resample_audio_frame(frame, sample_rate=self.sample_rate)
        self._samples = np.concatenate((self._samples, chunk))[-self.window_samples :]
        if self._samples.size < self.window_samples:
            output = np.pad(
                self._samples,
                (self.window_samples - self._samples.size, 0),
            )
        else:
            output = self._samples.copy()
        if self.peak_normalize:
            peak = float(np.max(np.abs(output)))
            if peak > 0.0:
                output /= peak
        return np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)


class RollingAudioWindow:
    """Accumulate audio until a full window is ready, then emit at a fixed hop.

    Unlike :class:`SlidingAudioWindow`, this class never pads an incomplete first
    window. It is suitable for slower generative models that should not be asked
    to caption mostly-silent synthetic padding at startup.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        window_seconds: float = 5.0,
        hop_seconds: float | None = None,
        peak_normalize: bool = False,
    ) -> None:
        if isinstance(sample_rate, bool) or sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        effective_hop = window_seconds if hop_seconds is None else hop_seconds
        if not 0.0 < effective_hop <= window_seconds:
            raise ValueError("hop_seconds must be greater than 0 and no greater than window_seconds")
        self.sample_rate = sample_rate
        self.window_samples = round(sample_rate * window_seconds)
        if self.window_samples < 1:
            raise ValueError("window_seconds must span at least one sample")
        self.hop_samples = round(sample_rate * effective_hop)
        if self.hop_samples < 1:
            raise ValueError("hop_seconds must span at least one sample")
        self.peak_normalize = peak_normalize
        self._samples = np.empty(0, dtype=np.float32)
        self._has_emitted = False
        self._samples_since_emit = 0

    def __call__(self, frame: AudioFrame) -> np.ndarray | None:
        chunk = resample_audio_frame(frame, sample_rate=self.sample_rate)
        self._samples = np.concatenate((self._samples, chunk))[-self.window_samples :]
        if self._samples.size < self.window_samples:
            return None
        if self._has_emitted:
            self._samples_since_emit += chunk.size
            if self._samples_since_emit < self.hop_samples:
                return None
        self._has_emitted = True
        self._samples_since_emit = 0
        output = self._samples.copy()
        if self.peak_normalize:
            peak = float(np.max(np.abs(output)))
            if peak > 0.0:
                output /= peak
        return np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)


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

    samples = resample_audio_frame(frame, sample_rate=sample_rate)

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
