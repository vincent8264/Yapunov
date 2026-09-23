"""Low-bandwidth, local-only audio display helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edge_ai.inputs.audio import AudioFrame


@dataclass
class AudioSpectrum:
    """Convert short audio frames into a 13-band equalizer-style display."""

    rate_hz: int = 20
    inference_duration_seconds: float = 1.0
    inference_hop_seconds: float | None = None
    floor_db: float = -60.0
    ceiling_db: float = -6.0
    _levels: np.ndarray = field(default_factory=lambda: np.zeros(13), init=False)
    _buffer: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False
    )
    _sample_rate: int | None = field(default=None, init=False)
    _samples_since_inference: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.rate_hz, bool)
            or not isinstance(self.rate_hz, int)
            or self.rate_hz < 1
        ):
            raise ValueError("audio spectrum rate_hz must be a positive integer")
        if (
            isinstance(self.inference_duration_seconds, bool)
            or not isinstance(self.inference_duration_seconds, (int, float))
            or self.inference_duration_seconds <= 0.0
        ):
            raise ValueError("audio spectrum inference duration must be positive")
        if self.inference_hop_seconds is not None and (
            isinstance(self.inference_hop_seconds, bool)
            or not isinstance(self.inference_hop_seconds, (int, float))
            or not 0.0 < self.inference_hop_seconds <= self.inference_duration_seconds
        ):
            raise ValueError(
                "audio spectrum inference hop must be positive and no longer than the window"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.floor_db, self.ceiling_db)
        ) or self.floor_db >= self.ceiling_db:
            raise ValueError("audio spectrum floor_db must be below ceiling_db")

    @property
    def frame_duration_seconds(self) -> float:
        return 1.0 / self.rate_hz

    def update(self, frame: AudioFrame) -> tuple[int, ...]:
        """Return 13 log-spaced spectral levels, with a short falling decay."""
        samples = frame.samples.astype(np.float64, copy=False)
        window = np.hanning(samples.size)
        spectrum = np.fft.rfft(samples * window)
        frequencies = np.fft.rfftfreq(samples.size, 1.0 / frame.sample_rate)
        # Use the audible range in logarithmic bands. The mean power per bin avoids
        # wider high-frequency bands appearing louder solely because they have more bins.
        upper_frequency = min(8_000.0, frame.sample_rate / 2.0)
        edges = np.geomspace(60.0, upper_frequency, 14)
        normalized_power = np.square(np.abs(spectrum) / (window.sum() / 2.0)) / 2.0
        targets = np.zeros(13)
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            in_band = (frequencies >= low) & (
                frequencies < high if index < 12 else frequencies <= high
            )
            if np.any(in_band):
                rms = float(np.sqrt(np.mean(normalized_power[in_band])))
                db = 20.0 * np.log10(max(rms, np.finfo(np.float64).tiny))
                targets[index] = np.clip(
                    (db - self.floor_db) / (self.ceiling_db - self.floor_db) * 8.0,
                    0.0,
                    8.0,
                )
        # Fast attack and a 40% per-frame fall make bars responsive without flicker.
        self._levels = np.maximum(targets, self._levels * 0.60)
        return tuple(int(value) for value in np.rint(self._levels))

    def append_for_inference(self, frame: AudioFrame) -> AudioFrame | None:
        """Accumulate display chunks and emit fixed-size audio inference windows.

        With no ``inference_hop_seconds``, windows are non-overlapping as before.
        A shorter hop retains the overlapping history and emits the newest window at
        that cadence after the initial window has filled.
        """
        if self._sample_rate is None:
            self._sample_rate = frame.sample_rate
        elif frame.sample_rate != self._sample_rate:
            raise ValueError("audio spectrum frames must keep a consistent sample rate")

        required = round(self._sample_rate * self.inference_duration_seconds)
        hop_seconds = self.inference_hop_seconds or self.inference_duration_seconds
        hop_samples = round(self._sample_rate * hop_seconds)
        if required < 1 or hop_samples < 1:
            raise ValueError(
                "audio spectrum inference window and hop must each span a sample"
            )
        was_full = self._buffer.size >= required
        self._buffer = np.concatenate((self._buffer, frame.samples))
        if self._buffer.size < required:
            return None
        self._buffer = self._buffer[-required:]
        # The first complete window runs immediately. Thereafter, each result
        # contains exactly one configured hop of new audio.
        if was_full:
            self._samples_since_inference += frame.samples.size
            if self._samples_since_inference < hop_samples:
                return None
        window = AudioFrame(self._buffer.copy(), self._sample_rate)
        self._samples_since_inference %= hop_samples
        return window
