from pathlib import Path
import wave

import numpy as np
import pytest

from edge_ai.inference.spectral import SpectralSoundInferenceEngine
from edge_ai.audio_display import AudioSpectrum
from edge_ai.inputs.audio import (
    ArduinoMicrophoneInput,
    AudioFrame,
    MicrophoneInput,
    SimulatedSoundInput,
    WavAudioInput,
)
from edge_ai.preprocessing.audio import extract_audio_features, prepare_audio_waveform


def test_waveform_resamples_and_pads_to_exact_shape() -> None:
    frame = AudioFrame(np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32), 4)

    waveform = prepare_audio_waveform(frame, sample_rate=8, duration_seconds=1.0)

    assert waveform.dtype == np.float32
    assert waveform.shape == (8,)
    assert np.all(np.isfinite(waveform))


def test_downsampling_preserves_passband_and_rejects_aliases() -> None:
    sample_rate = 48_000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    passband = AudioFrame(np.sin(2 * np.pi * 1_000 * time_axis), sample_rate)
    stopband = AudioFrame(np.sin(2 * np.pi * 12_000 * time_axis), sample_rate)

    passed = prepare_audio_waveform(passband, sample_rate=16_000, duration_seconds=1.0)
    rejected = prepare_audio_waveform(stopband, sample_rate=16_000, duration_seconds=1.0)

    # Ignore the short filter transients at the boundaries.
    assert np.sqrt(np.mean(np.square(passed[64:-64]))) == pytest.approx(
        1 / np.sqrt(2), rel=0.02
    )
    assert np.sqrt(np.mean(np.square(rejected[64:-64]))) < 0.01


def test_wav_input_downmixes_stereo_pcm(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    stereo = np.array([[32767, -32768], [16384, 16384]], dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(stereo.tobytes())

    frame = WavAudioInput([path], loop=False).read()

    assert frame.sample_rate == 8_000
    assert frame.samples.shape == (2,)
    assert frame.samples[0] == pytest.approx(-1.0 / 65536.0)
    assert frame.samples[1] == pytest.approx(0.5)


class _FakeArduinoMicrophone:
    def __init__(self, chunks: list[np.ndarray | None], *, sample_rate: int) -> None:
        self.chunks = list(chunks)
        self.sample_rate = sample_rate
        self.channels = 1
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def capture(self) -> np.ndarray | None:
        return self.chunks.pop(0) if self.chunks else None

    def stop(self) -> None:
        self.stopped = True


def test_arduino_microphone_assembles_int16_chunks_into_frames() -> None:
    chunk = np.full(3, 16384, dtype=np.int16)
    fake = _FakeArduinoMicrophone([chunk, None, chunk, chunk], sample_rate=8)
    requested: dict[str, object] = {}

    def factory(**kwargs: object) -> _FakeArduinoMicrophone:
        requested.update(kwargs)
        return fake

    source = ArduinoMicrophoneInput(
        sample_rate=8, duration_seconds=0.5, device=0, microphone_factory=factory
    )
    first = source.read()
    second = source.read()
    source.close()

    assert requested == {"device": 0, "sample_rate": 8}
    assert fake.started and fake.stopped
    assert first.sample_rate == 8
    assert first.samples.shape == (4,)
    assert np.allclose(first.samples, 0.5)
    assert second.samples.shape == (4,)


def test_arduino_microphone_uses_the_rate_the_device_opened() -> None:
    fake = _FakeArduinoMicrophone([np.zeros(48, dtype=np.int16)], sample_rate=48)

    source = ArduinoMicrophoneInput(
        sample_rate=16, duration_seconds=1.0, microphone_factory=lambda **_: fake
    )

    assert source.sample_rate == 48
    assert source.read().samples.shape == (48,)


def test_arduino_microphone_times_out_without_audio() -> None:
    fake = _FakeArduinoMicrophone([], sample_rate=16)
    source = ArduinoMicrophoneInput(
        sample_rate=16,
        duration_seconds=1.0,
        read_timeout_seconds=0.02,
        microphone_factory=lambda **_: fake,
    )

    with pytest.raises(RuntimeError, match="no audio"):
        source.read()


def test_arduino_microphone_requires_app_lab_runtime() -> None:
    with pytest.raises(RuntimeError, match="App Lab runtime"):
        ArduinoMicrophoneInput()


def test_random_simulated_audio_stays_within_requested_events() -> None:
    events = ["smoke_alarm", "glass_break", "fall_thud"]
    source = SimulatedSoundInput(events, seed=11, choose_randomly=True)
    engine = SpectralSoundInferenceEngine()

    labels = [
        engine.predict(extract_audio_features(source.read())).label for _ in range(12)
    ]

    assert set(labels) <= set(events)
    assert len(set(labels)) > 1


def test_simulated_audio_exercises_all_demo_classes() -> None:
    events = ["background", "smoke_alarm", "glass_break", "fall_thud"]
    source = SimulatedSoundInput(events, loop=False)
    engine = SpectralSoundInferenceEngine()

    labels = [engine.predict(extract_audio_features(source.read())).label for _ in events]

    assert labels == events


def test_audio_frame_rejects_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        AudioFrame(np.array([np.nan], dtype=np.float32), 16_000)


def test_audio_spectrum_separates_bands_and_reassembles_model_window() -> None:
    spectrum = AudioSpectrum(rate_hz=20, inference_duration_seconds=1.0)
    samples = np.arange(800, dtype=np.float32) / 16_000
    quiet = AudioFrame(np.zeros(800, dtype=np.float32), 16_000)
    tone = AudioFrame(np.sin(2 * np.pi * 2_000 * samples), 16_000)

    assert spectrum.update(quiet) == (0,) * 13
    columns = spectrum.update(tone)
    assert max(columns) > 0
    assert columns.index(max(columns)) > 6

    assert spectrum.append_for_inference(quiet) is None
    for _ in range(18):
        assert spectrum.append_for_inference(quiet) is None
    window = spectrum.append_for_inference(tone)

    assert window is not None
    assert window.samples.shape == (16_000,)
    assert window.samples[-800:].tolist() == tone.samples.tolist()


def test_microphone_reads_and_closes_injected_stream() -> None:
    class Stream:
        stopped = False
        closed = False

        def start(self) -> None:
            pass

        def read(self, frame_count: int) -> tuple[np.ndarray, bool]:
            return np.zeros((frame_count, 1), dtype=np.float32), False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    stream = Stream()

    class Backend:
        @staticmethod
        def InputStream(**kwargs: object) -> Stream:
            assert kwargs["device"] == "MOVO USB-M1"
            return stream

    source = MicrophoneInput(
        sample_rate=8_000,
        duration_seconds=0.25,
        device="MOVO USB-M1",
        backend=Backend,
    )
    frame = source.read()
    source.close()

    assert frame.samples.shape == (2_000,)
    assert stream.stopped is True
    assert stream.closed is True


def test_microphone_falls_back_to_callback_stream_when_blocking_is_unsupported() -> None:
    class Stream:
        stopped = False
        closed = False

        def __init__(self, callback: object) -> None:
            self.callback = callback

        def start(self) -> None:
            self.callback(np.full((4, 1), 0.25, dtype=np.float32), 4, None, None)

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    class Backend:
        stream: Stream | None = None

        @classmethod
        def InputStream(cls, **kwargs: object) -> Stream:
            callback = kwargs.get("callback")
            if callback is None:
                raise RuntimeError("blocking API unsupported")
            cls.stream = Stream(callback)
            return cls.stream

    source = MicrophoneInput(
        sample_rate=8,
        duration_seconds=0.5,
        device=31,
        backend=Backend,
    )

    assert np.all(source.read().samples == 0.25)
    source.close()
    assert Backend.stream is not None
    assert Backend.stream.stopped is True
    assert Backend.stream.closed is True
