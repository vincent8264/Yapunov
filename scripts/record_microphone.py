"""Record the default microphone to a PCM WAV file.

Run with ``uv run python scripts/record_microphone.py``. Press Enter to begin
recording, then Ctrl+C to finish the recording and save the WAV file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import wave


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / "recordings" / f"recording-{timestamp}.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a microphone until Ctrl+C, then save a WAV file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="destination WAV path (default: runs/recordings/recording-<timestamp>.wav)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="samples per second (default: 16000, suitable for YAMNet)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice microphone name or numeric device index",
    )
    args = parser.parse_args()
    if args.device is not None and args.device.isdecimal():
        args.device = int(args.device)
    return args


def record(output_path: Path, *, sample_rate: int, device: str | int | None) -> None:
    """Capture mono 16-bit PCM until the caller interrupts the process."""
    if sample_rate < 1:
        raise ValueError("sample rate must be positive")
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("recording requires the project's sounddevice dependency") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            def write_audio(indata: bytes, _frames: int, _time: object, status: object) -> None:
                if status:
                    print(f"Microphone status: {status}", file=sys.stderr)
                wav_file.writeframesraw(indata)

            with sd.RawInputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=device,
                callback=write_audio,
            ):
                print("Recording. Press Ctrl+C to stop and save...")
                while True:
                    time.sleep(0.25)
    except KeyboardInterrupt:
        print(f"Saved recording to {output_path}")


def main() -> int:
    args = parse_args()
    print("Press Enter to start recording (Ctrl+C afterwards saves the WAV file).")
    input()
    try:
        record(args.output, sample_rate=args.sample_rate, device=args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"Could not record audio: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
