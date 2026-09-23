"""Audio inputs for deterministic replay, live microphones, and laptop demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import queue
import random
import time
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


class MicrophoneHealthError(RuntimeError):
    """A locally observed microphone failure or persistently unusable signal."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class MicrophoneHealthInput(InputSource):
    """Validate microphone frames without claiming a specific physical failure.

    A missing stream, digital silence, and a frozen capture buffer can look alike from
    software.  This wrapper therefore reports the observation and possible causes
    rather than asserting that the microphone hardware is broken.
    """

    def __init__(
        self,
        source: InputSource | None,
        *,
        source_factory: Callable[[], InputSource] | None = None,
        reopen_on_fault: bool = True,
        initial_error: Exception | None = None,
        failure_seconds: float = 5.0,
        silence_threshold: float = 0.0,
        detect_frozen: bool = True,
        retry_interval_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_seconds <= 0.0:
            raise ValueError("microphone health failure_seconds must be positive")
        if not 0.0 <= silence_threshold <= 1.0:
            raise ValueError("microphone health silence_threshold must be between 0 and 1")
        if retry_interval_seconds <= 0.0:
            raise ValueError("microphone health retry_interval_seconds must be positive")
        if source is None and source_factory is None:
            raise ValueError("microphone health requires a source or source_factory")
        self.source: InputSource | None = source
        self.source_factory = source_factory
        self.reopen_on_fault = reopen_on_fault
        self.failure_seconds = failure_seconds
        self.silence_threshold = silence_threshold
        self.detect_frozen = detect_frozen
        self.retry_interval_seconds = retry_interval_seconds
        self._clock = clock
        self._silent_seconds = 0.0
        self._frozen_seconds = 0.0
        self._previous: AudioFrame | None = None
        self._fault: MicrophoneHealthError | None = (
            MicrophoneHealthError(
                "unavailable",
                f"microphone open failed: {type(initial_error).__name__}: {initial_error}",
            )
            if initial_error is not None
            else None
        )
        self._recovered_reason: str | None = None
        self._next_retry_at = (
            self._clock() + self.retry_interval_seconds
            if initial_error is not None
            else 0.0
        )

    def _close_source(self, *, suppress_errors: bool) -> None:
        if self.source is None:
            return
        close = getattr(self.source, "close", None)
        try:
            if callable(close):
                close()
        except Exception:
            # A disconnected capture device can also fail during close. The read
            # failure remains the useful diagnosis and retries must still continue.
            if not suppress_errors:
                raise
        finally:
            self.source = None

    def _mark_fault(self, reason: str, message: str) -> MicrophoneHealthError:
        fault = MicrophoneHealthError(reason, message)
        if self._fault is None:
            self._fault = fault
        self._silent_seconds = 0.0
        self._frozen_seconds = 0.0
        self._previous = None
        if self.source_factory is not None and (
            self.source is None or self.reopen_on_fault
        ):
            self._next_retry_at = self._clock() + self.retry_interval_seconds
        return fault

    def _ensure_source(self) -> InputSource:
        if (
            self._fault is not None
            and self.source_factory is not None
            and (self.source is None or self.reopen_on_fault)
        ):
            if self._clock() < self._next_retry_at:
                raise self._fault
            # Report the fault before touching a disconnected peripheral. Some
            # board runtimes can block in stop() after USB removal, so cleanup is
            # intentionally deferred until the first reconnect attempt.
            self._close_source(suppress_errors=True)
            try:
                self.source = self.source_factory()
            except Exception as exc:
                self._next_retry_at = self._clock() + self.retry_interval_seconds
                raise MicrophoneHealthError(
                    "unavailable",
                    f"microphone reconnect failed: {type(exc).__name__}: {exc}",
                ) from exc
            return self.source
        if self.source is not None:
            return self.source
        if self.source_factory is None or self._fault is None:
            raise RuntimeError("microphone input is closed")
        raise self._fault

    def take_recovered_reason(self) -> str | None:
        """Return one recovery transition after a healthy frame resumes."""
        reason, self._recovered_reason = self._recovered_reason, None
        return reason

    def read(self) -> AudioFrame:
        try:
            frame = self._ensure_source().read()
        except MicrophoneHealthError:
            raise
        except Exception as exc:
            raise self._mark_fault(
                "unavailable", f"microphone read failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(frame, AudioFrame):
            raise self._mark_fault(
                "invalid_audio",
                f"microphone returned {type(frame).__name__} instead of an AudioFrame",
            )

        duration_seconds = frame.samples.size / frame.sample_rate
        peak = float(np.max(np.abs(frame.samples)))
        self._silent_seconds = (
            self._silent_seconds + duration_seconds
            if peak <= self.silence_threshold
            else 0.0
        )

        repeated = (
            self.detect_frozen
            and self._previous is not None
            and frame.sample_rate == self._previous.sample_rate
            and np.array_equal(frame.samples, self._previous.samples)
        )
        previous = self._previous
        self._frozen_seconds = self._frozen_seconds + duration_seconds if repeated else 0.0
        self._previous = frame

        if self._fault is not None:
            usable_signal = peak > self.silence_threshold
            changing_signal = previous is not None and not repeated
            if not usable_signal or (
                self._fault.reason == "frozen_signal" and not changing_signal
            ):
                raise self._fault
            self._recovered_reason = self._fault.reason
            self._fault = None
            self._silent_seconds = 0.0
            self._frozen_seconds = 0.0
            return frame

        if self._silent_seconds >= self.failure_seconds:
            raise self._mark_fault(
                "no_signal",
                f"microphone produced no signal for {self._silent_seconds:.1f} seconds; "
                "it may be muted, disconnected, or unavailable",
            )
        if self._frozen_seconds >= self.failure_seconds:
            raise self._mark_fault(
                "frozen_signal",
                f"microphone repeated an identical audio buffer for "
                f"{self._frozen_seconds:.1f} seconds; capture may be stalled",
            )
        return frame

    def close(self) -> None:
        self.source_factory = None
        if self._fault is not None and not self.reopen_on_fault:
            # App Lab's stop() can block after USB removal. Let container teardown
            # release the invalid ALSA handle instead of trapping the Python process
            # in cleanup and leaving the next app unable to claim the microphone.
            return
        self._close_source(suppress_errors=False)


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
        self._callback_queue: queue.Queue[np.ndarray] | None = None
        self._callback_samples = np.empty(0, dtype=np.float32)
        self._callback_stream = False
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
            # Some Windows WDM-KS devices do not implement SoundDevice's blocking
            # read API but do accept an input callback. Keep both paths so ordinary
            # devices and injected test streams remain simple.
            self._callback_queue = queue.Queue()
            try:
                self._stream = backend.InputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.frame_count,
                    device=device,
                    callback=self._capture_callback,
                )
                self._stream.start()
                self._callback_stream = True
            except Exception as callback_exc:
                raise RuntimeError(
                    f"could not open microphone {device!r}: {callback_exc} "
                    f"(blocking stream also failed: {exc})"
                ) from callback_exc

    def _capture_callback(self, samples: np.ndarray, frames: int, *_: object) -> None:
        if self._callback_queue is None:
            return
        captured = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if captured.size:
            self._callback_queue.put_nowait(captured)

    def read(self) -> AudioFrame:
        if self._callback_stream:
            assert self._callback_queue is not None
            while self._callback_samples.size < self.frame_count:
                try:
                    next_samples = self._callback_queue.get(timeout=3.0)
                except queue.Empty as exc:
                    raise RuntimeError("microphone callback timed out") from exc
                self._callback_samples = np.concatenate(
                    (self._callback_samples, next_samples)
                )
            samples = self._callback_samples[: self.frame_count]
            self._callback_samples = self._callback_samples[self.frame_count :]
            return AudioFrame(samples, self.sample_rate)
        samples, overflowed = self._stream.read(self.frame_count)
        if overflowed:
            raise RuntimeError("microphone input overflowed; audio samples were lost")
        return AudioFrame(np.asarray(samples, dtype=np.float32).reshape(-1), self.sample_rate)

    def close(self) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()


def _pcm_to_float(samples: np.ndarray) -> np.ndarray:
    """Convert integer or float PCM samples to mono float32 in roughly ``[-1, 1]``."""
    samples = np.asarray(samples)
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    if samples.dtype == np.int32:
        return samples.astype(np.float32) / 2147483648.0
    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32)
    raise RuntimeError(f"unsupported microphone sample format: {samples.dtype}")


class ArduinoMicrophoneInput(InputSource):
    """Capture fixed-size frames through Arduino App Lab's ALSA microphone peripheral.

    ``arduino.app_peripherals.microphone`` exists only inside the App Lab runtime on
    the UNO Q, so it is imported when the input is constructed.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        duration_seconds: float = 1.0,
        device: str | int | None = None,
        read_timeout_seconds: float = 3.0,
        microphone_factory: Any | None = None,
    ) -> None:
        if isinstance(sample_rate, bool) or sample_rate < 1:
            raise ValueError("microphone sample_rate must be positive")
        if duration_seconds <= 0.0:
            raise ValueError("microphone duration_seconds must be positive")
        if read_timeout_seconds <= 0.0:
            raise ValueError("microphone read timeout must be positive")
        if microphone_factory is None:
            try:
                from arduino.app_peripherals.microphone import Microphone
            except ImportError as exc:
                raise RuntimeError(
                    "arduino_microphone input requires the Arduino App Lab runtime on "
                    "the UNO Q; use type = 'microphone' on a laptop"
                ) from exc
            microphone_factory = Microphone
        self.duration_seconds = duration_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self._pending = np.empty(0, dtype=np.float32)
        try:
            self._microphone = microphone_factory(device=device, sample_rate=sample_rate)
            self._microphone.start()
        except Exception as exc:
            raise RuntimeError(f"could not open App Lab microphone {device!r}: {exc}") from exc
        # The device may not support the requested rate; the peripheral reports the
        # rate it actually opened, and preprocessing resamples from it.
        self.sample_rate = int(getattr(self._microphone, "sample_rate", sample_rate))
        self.channels = int(getattr(self._microphone, "channels", 1))
        self.frame_count = round(self.sample_rate * duration_seconds)

    def read(self) -> AudioFrame:
        deadline = time.monotonic() + self.read_timeout_seconds
        chunks = [self._pending]
        collected = self._pending.size
        while collected < self.frame_count:
            chunk = self._microphone.capture()
            if chunk is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("App Lab microphone returned no audio before the timeout")
                time.sleep(0.005)
                continue
            # App Lab microphone implementations may return mono audio as either
            # ``[frames]`` or ``[frames, 1]``. Flatten before buffering so both forms,
            # along with interleaved multi-channel PCM, follow the same path.
            samples = _pcm_to_float(chunk).reshape(-1)
            if self.channels > 1:
                samples = samples[: samples.size - samples.size % self.channels]
                samples = samples.reshape(-1, self.channels).mean(axis=1)
            if samples.size == 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("App Lab microphone returned no audio before the timeout")
                time.sleep(0.005)
                continue
            chunks.append(samples)
            collected += samples.size
            deadline = time.monotonic() + self.read_timeout_seconds
        buffered = np.concatenate(chunks)
        self._pending = buffered[self.frame_count :]
        return AudioFrame(buffered[: self.frame_count], self.sample_rate)

    def close(self) -> None:
        self._microphone.stop()


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
        choose_randomly: bool = False,
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
        self.choose_randomly = choose_randomly
        self._index = 0
        self._random = random.Random(seed)

    def read(self) -> AudioFrame:
        if self.choose_randomly:
            event = self._random.choice(self.events)
        else:
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
            samples = self._fall_thud(generator)

        peak = max(float(np.max(np.abs(samples))), 1.0)
        return AudioFrame((samples / peak).astype(np.float32), self.sample_rate)

    def _fall_thud(self, generator: np.random.Generator) -> np.ndarray:
        """Return a body-impact-like thud plus a smaller bounce.

        A pure low sine is heard by YAMNet as silence, so each impact combines a
        short click, low-passed noise, and a 77 Hz body resonance.
        """
        samples = np.zeros(self.sample_count)
        for start_seconds, amplitude in ((0.15, 1.0), (0.42, 0.67)):
            start = int(start_seconds * self.sample_rate)
            if start >= self.sample_count:
                break
            count = self.sample_count - start
            elapsed = np.arange(count) / self.sample_rate
            spectrum = np.fft.rfft(generator.normal(0.0, 1.0, count))
            spectrum[np.fft.rfftfreq(count, 1.0 / self.sample_rate) > 1_700.0] = 0.0
            low_noise = np.fft.irfft(spectrum, count)
            low_noise /= max(float(np.max(np.abs(low_noise))), 1e-9)
            body = 0.9 * np.sin(2.0 * np.pi * 77.0 * elapsed)
            envelope = np.exp(-9.7 * elapsed) * (1.0 - np.exp(-elapsed / 0.01))
            click = generator.normal(0.0, 1.0, count) * np.exp(-340.0 * elapsed)
            samples[start:] += amplitude * ((low_noise + body) * envelope + click)
        return 0.9 * samples / max(float(np.max(np.abs(samples))), 1e-9)
