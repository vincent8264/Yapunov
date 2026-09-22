"""Command-line entry point for the Edge AI starter."""

import argparse
from pathlib import Path
from typing import Sequence

from edge_ai.config import ConfigError, load_config, load_hardware_check_config
from edge_ai.diagnostics import run_hardware_checks
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            configured = load_config(args.config)
            run_pipeline(configured, max_steps=args.max_steps)
        else:
            plan = load_hardware_check_config(args.config)
            result = run_hardware_checks(plan, allow_actuators=args.allow_actuators)
            for check in result.checks:
                print(f"{check.status.upper():7} {check.name}: {check.detail}")
            if not result.passed:
                return 1
    except (ConfigError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
