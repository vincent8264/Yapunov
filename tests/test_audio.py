from pathlib import Path
import wave

import numpy as np
import pytest

from edge_ai.inference.spectral import SpectralSoundInferenceEngine
from edge_ai.inputs.audio import AudioFrame, MicrophoneInput, SimulatedSoundInput, WavAudioInput
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


def test_simulated_audio_exercises_all_demo_classes() -> None:
    events = ["background", "smoke_alarm", "glass_break", "fall_thud"]
    source = SimulatedSoundInput(events, loop=False)
    engine = SpectralSoundInferenceEngine()

    labels = [engine.predict(extract_audio_features(source.read())).label for _ in events]

    assert labels == events


def test_audio_frame_rejects_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        AudioFrame(np.array([np.nan], dtype=np.float32), 16_000)


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
