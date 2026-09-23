"""Build a small real-speech help/background evaluation set from LibriSpeech."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re


HELP = re.compile(r"\bHELP\b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/help-kws/real"))
    parser.add_argument(
        "--background-output",
        type=Path,
        default=Path("data/help-kws/librispeech-background"),
    )
    parser.add_argument(
        "--training-help-output",
        type=Path,
        help="optional folder for approximately aligned one-second help clips",
    )
    parser.add_argument("--test-background-count", type=int, default=60)
    parser.add_argument("--train-background-count", type=int, default=300)
    parser.add_argument("--validation-background-count", type=int, default=60)
    args = parser.parse_args()

    import soundfile as sf

    help_ids: list[str] = []
    background_ids: list[str] = []
    locations: dict[str, Path] = {}
    transcripts: dict[str, str] = {}
    for transcript in sorted(args.librispeech.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            utterance_id, text = line.split(" ", 1)
            locations[utterance_id] = transcript.parent / f"{utterance_id}.flac"
            transcripts[utterance_id] = text
            if HELP.search(text):
                help_ids.append(utterance_id)
            else:
                background_ids.append(utterance_id)

    help_speakers = {identifier.split("-", 1)[0] for identifier in help_ids}
    background_ids = [
        identifier
        for identifier in background_ids
        if identifier.split("-", 1)[0] not in help_speakers
    ]
    random.Random(8264).shuffle(background_ids)
    total_background = (
        args.test_background_count
        + args.train_background_count
        + args.validation_background_count
    )
    if len(background_ids) < total_background:
        raise ValueError(
            f"requested {total_background} background clips, found {len(background_ids)}"
        )
    test_background = background_ids[: args.test_background_count]
    train_start = args.test_background_count
    train_background = background_ids[
        train_start : train_start + args.train_background_count
    ]
    validation_background = background_ids[
        train_start + args.train_background_count : total_background
    ]

    for folder, identifiers in (
        ("help_call", help_ids),
        ("background", test_background),
    ):
        destination = args.output / folder
        destination.mkdir(parents=True, exist_ok=True)
        for utterance_id in identifiers:
            samples, sample_rate = sf.read(
                locations[utterance_id], dtype="float32", always_2d=False
            )
            sf.write(
                destination / f"{utterance_id}.wav",
                samples,
                sample_rate,
                subtype="PCM_16",
            )

    for split, identifiers in (
        ("train", train_background),
        ("validation", validation_background),
    ):
        destination = args.background_output / split
        destination.mkdir(parents=True, exist_ok=True)
        for utterance_id in identifiers:
            samples, sample_rate = sf.read(
                locations[utterance_id], dtype="float32", always_2d=False
            )
            sf.write(
                destination / f"{utterance_id}.wav",
                samples,
                sample_rate,
                subtype="PCM_16",
            )

    if args.training_help_output is not None:
        import numpy as np

        args.training_help_output.mkdir(parents=True, exist_ok=True)
        for utterance_id in help_ids:
            samples, sample_rate = sf.read(
                locations[utterance_id], dtype="float32", always_2d=False
            )
            words = transcripts[utterance_id].split()
            word_index = words.index("HELP")
            center = round((word_index + 0.5) / len(words) * samples.size)
            window_size = sample_rate
            start = min(max(0, center - window_size // 2), max(0, samples.size - window_size))
            window = samples[start : start + window_size]
            if window.size < window_size:
                window = np.pad(window, (0, window_size - window.size))
            sf.write(
                args.training_help_output / f"{utterance_id}.wav",
                window,
                sample_rate,
                subtype="PCM_16",
            )

    print(
        f"wrote test={len(help_ids)} help/{len(test_background)} background, "
        f"train={len(train_background)} background, and "
        f"validation={len(validation_background)} background clips"
    )
    if args.training_help_output is not None:
        print(f"wrote {len(help_ids)} approximate help windows to training output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
