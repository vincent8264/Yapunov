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
from edge_ai.notifications import AlertNotification
from edge_ai.runner import run_pipeline


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
        choices=("smoke_alarm", "glass_break", "fall_thud"),
        default="smoke_alarm",
        help="preset alert to send (default: smoke_alarm)",
    )
    test_notification.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="test classifier confidence from 0 to 1 (default: 0.95)",
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
        else:
            _send_test_notification(args.config, args.event, args.confidence)
    except (ConfigError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
