"""Command-line entry point for the Edge AI starter."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from edge_ai.config import (
    ConfigError,
    load_config,
    load_hardware_check_config,
    load_notification_config,
)
from edge_ai.diagnostics import run_hardware_checks
from edge_ai.evaluation import evaluate_keyword_dataset
from edge_ai.inference.keyword import KeywordSpotterInferenceEngine
from edge_ai.inference.yamnet import YAMNetInferenceEngine
from edge_ai.inputs.audio import read_wav
from edge_ai.notifications import AlertNotification
from edge_ai.preprocessing.audio import resample_audio_frame
from edge_ai.runner import run_pipeline
from edge_ai.setup_server import run_setup_server
from edge_ai.transcription import SherpaZipformerTranscriber
from edge_ai.inputs.audio import MicrophoneInput


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a configured pipeline")
    run.add_argument("--config", type=Path, required=True, help="path to a TOML config")
    run.add_argument("--max-steps", type=int, help="stop after this many steps")
    hardware_check = subparsers.add_parser(
        "hardware-check", help="run safe configured hardware diagnostics"
    )
    hardware_check.add_argument(
        "--config", type=Path, required=True, help="path to a hardware-check TOML config"
    )
    hardware_check.add_argument(
        "--allow-actuators",
        action="store_true",
        help="allow configured PWM and servo movement",
    )
    test_notification = subparsers.add_parser(
        "test-notification", help="send one test alert using configured notifications"
    )
    test_notification.add_argument(
        "--config", type=Path, required=True, help="path to a TOML config"
    )
    test_notification.add_argument(
        "--event",
        choices=("smoke_alarm", "glass_break", "fall_thud", "help_call"),
        default="smoke_alarm",
        help="preset alert to send (default: smoke_alarm)",
    )
    test_notification.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="test classifier confidence from 0 to 1 (default: 0.95)",
    )
    setup = subparsers.add_parser(
        "setup", help="serve the temporary local notification setup portal"
    )
    setup.add_argument("--config", type=Path, required=True, help="path to a TOML config")
    setup.add_argument(
        "--host",
        default="127.0.0.1",
        help="listen address; use 0.0.0.0 for access over the device LAN",
    )
    setup.add_argument("--port", type=int, default=8080, help="listen port (default: 8080)")
    inspect_audio = subparsers.add_parser(
        "inspect-audio", help="show raw YAMNet AudioSet predictions for one WAV file"
    )
    inspect_audio.add_argument("audio", type=Path, help="PCM WAV file to inspect")
    inspect_audio.add_argument("--model", type=Path, required=True, help="YAMNet ONNX model")
    inspect_audio.add_argument(
        "--class-map", type=Path, required=True, help="YAMNet class-map CSV"
    )
    inspect_audio.add_argument(
        "--top-k", type=int, default=10, help="number of raw classes to print"
    )
    evaluate_keyword = subparsers.add_parser(
        "evaluate-keyword", help="evaluate a help keyword model on labeled WAV clips"
    )
    evaluate_keyword.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="folder containing help_call/ and background/ WAV files",
    )
    evaluate_keyword.add_argument(
        "--model", type=Path, required=True, help="help keyword ONNX model"
    )
    evaluate_keyword.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="help probability threshold (default: 0.5)",
    )
    evaluate_keyword.add_argument(
        "--positive-index",
        type=int,
        default=1,
        help="help class index for multi-score outputs (default: 1)",
    )
    transcribe = subparsers.add_parser(
        "transcribe", help="transcribe a local microphone with streaming Sherpa ASR"
    )
    transcribe.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="unpacked sherpa-onnx-streaming-zipformer-en-2023-06-26 directory",
    )
    transcribe.add_argument(
        "--device",
        help="microphone name or numeric SoundDevice index (default: system default)",
    )
    transcribe.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.2,
        help="microphone frame duration in seconds (default: 0.2)",
    )
    transcribe.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ASR CPU threads; keep at 1 when sharing YAMNet (default: 1)",
    )
    return parser


def _send_test_notification(config: Path, event: str, confidence: float) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ConfigError("--confidence must be between 0 and 1")
    configured = load_notification_config(config)
    if configured.notifier is None:
        raise ConfigError("[notifications].type must be configured for delivery, not 'none'")
    alert = AlertNotification(
        event=event,
        confidence=confidence,
        device_name=configured.device_name,
        timestamp=datetime.now(configured.local_timezone).isoformat(timespec="seconds"),
    )
    try:
        configured.notifier.notify(alert)
    except Exception as exc:
        raise RuntimeError(
            f"test notification failed via {configured.notifier.channel}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        configured.notifier.close()
    print(f"Test notification delivered: channel={configured.notifier.channel} event={event}")


def _inspect_audio(audio: Path, model: Path, class_map: Path, top_k: int) -> None:
    frame = read_wav(audio)
    waveform = resample_audio_frame(frame, sample_rate=16_000)
    engine = YAMNetInferenceEngine(model, class_map_path=class_map)
    for rank, (label, confidence) in enumerate(
        engine.top_classes(waveform, limit=top_k), start=1
    ):
        print(f"{rank:2d}. {label:<40} {confidence:.3f}")


def _evaluate_keyword(
    dataset: Path,
    model: Path,
    threshold: float,
    positive_index: int,
) -> None:
    engine = KeywordSpotterInferenceEngine(
        model,
        activation_threshold=threshold,
        positive_index=positive_index,
    )


def _transcribe(
    model_dir: Path,
    device: str | None,
    chunk_seconds: float,
    threads: int,
) -> None:
    if chunk_seconds <= 0.0:
        raise ValueError("--chunk-seconds must be positive")
    transcriber = SherpaZipformerTranscriber(model_dir, num_threads=threads)
    microphone = MicrophoneInput(
        sample_rate=16_000,
        duration_seconds=chunk_seconds,
        device=int(device) if device is not None and device.isdigit() else device,
    )
    print("Listening. Press Ctrl+C to stop.")
    try:
        while True:
            for update in transcriber.accept(microphone.read()):
                prefix = "final" if update.is_final else "partial"
                print(f"{prefix}: {update.text}")
    finally:
        try:
            update = transcriber.finish()
            if update is not None:
                print(f"final: {update.text}")
        finally:
            microphone.close()
    result = evaluate_keyword_dataset(dataset, engine)
    print(
        f"clips={result.total} TP={result.true_positive} FP={result.false_positive} "
        f"TN={result.true_negative} FN={result.false_negative}"
    )
    print(
        f"precision={result.precision:.3f} recall={result.recall:.3f} "
        f"accuracy={result.accuracy:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            configured = load_config(args.config)
            run_pipeline(configured, max_steps=args.max_steps)
        elif args.command == "hardware-check":
            plan = load_hardware_check_config(args.config)
            result = run_hardware_checks(plan, allow_actuators=args.allow_actuators)
            for check in result.checks:
                print(f"{check.status.upper():7} {check.name}: {check.detail}")
            if not result.passed:
                return 1
        elif args.command == "test-notification":
            _send_test_notification(args.config, args.event, args.confidence)
        elif args.command == "setup":
            run_setup_server(args.config, host=args.host, port=args.port)
        elif args.command == "inspect-audio":
            _inspect_audio(args.audio, args.model, args.class_map, args.top_k)
        elif args.command == "transcribe":
            _transcribe(args.model_dir, args.device, args.chunk_seconds, args.threads)
        else:
            _evaluate_keyword(
                args.dataset,
                args.model,
                args.threshold,
                args.positive_index,
            )
    except (ConfigError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
