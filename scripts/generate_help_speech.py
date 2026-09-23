"""Generate a speaker-disjoint synthetic help-keyword dataset with Piper."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import wave


POSITIVE = ("help", "help me", "please help", "somebody help")
HARD_NEGATIVE = (
    "hello",
    "health",
    "shelf",
    "held",
    "helm",
    "yelp",
    "helpful",
    "I need assistance",
    "please call someone",
    "is anybody there",
)


def _synthesize_set(
    voice: object,
    directory: Path,
    phrases: tuple[str, ...],
    speakers: list[int],
    seed: int,
) -> None:
    from piper.config import SynthesisConfig

    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for index, speaker in enumerate(speakers):
        config = SynthesisConfig(
            speaker_id=speaker,
            length_scale=rng.uniform(0.82, 1.18),
            noise_scale=rng.uniform(0.55, 0.75),
            noise_w_scale=rng.uniform(0.65, 0.90),
            volume=rng.uniform(0.65, 1.0),
        )
        path = directory / f"speaker-{speaker:03d}-{index:03d}.wav"
        with wave.open(str(path), "wb") as output:
            voice.synthesize_wav(phrases[index % len(phrases)], output, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, required=True, help="Piper voice ONNX file")
    parser.add_argument("--output", type=Path, default=Path("data/help-kws/synthetic"))
    parser.add_argument("--train-speakers", type=int, default=180)
    parser.add_argument("--validation-speakers", type=int, default=60)
    args = parser.parse_args()

    from piper.voice import PiperVoice

    voice = PiperVoice.load(args.voice)
    total = args.train_speakers + args.validation_speakers
    if total > voice.config.num_speakers:
        raise ValueError(
            f"requested {total} speakers but voice contains {voice.config.num_speakers}"
        )
    speakers = list(range(voice.config.num_speakers))
    random.Random(8264).shuffle(speakers)
    train = speakers[: args.train_speakers]
    validation = speakers[args.train_speakers : total]
    _synthesize_set(voice, args.output / "train/help_call", POSITIVE, train, 1)
    _synthesize_set(voice, args.output / "train/background", HARD_NEGATIVE, train, 2)
    _synthesize_set(voice, args.output / "validation/help_call", POSITIVE, validation, 3)
    _synthesize_set(
        voice, args.output / "validation/background", HARD_NEGATIVE, validation, 4
    )
    print(f"wrote {2 * total} clips to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
