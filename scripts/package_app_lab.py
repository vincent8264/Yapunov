"""Build an Arduino App Lab zip that contains the sound pipeline.

Pass ``--notifications`` with a private SMTP config to enable email on the board.
The SMTP password is read from that config's ``password_env`` variable and written
only to ``python/smtp-password`` inside the git-ignored ``dist/`` output.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tomllib
import wave
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = REPO_ROOT / "app_lab" / "starter_app"
PACKAGE_SOURCE = REPO_ROOT / "src" / "edge_ai"
CONFIG_SOURCE = REPO_ROOT / "configs" / "sound-uno-q.toml"
YAMNET_CONFIG_SOURCE = REPO_ROOT / "configs" / "sound-yamnet-uno-q.toml"
YAMNET_MODEL_SOURCE = REPO_ROOT / "models" / "yamnet.onnx"
YAMNET_CLASS_MAP_SOURCE = REPO_ROOT / "models" / "yamnet_class_map.csv"
BOARD_CONFIG_NAME = "sound-uno-q.toml"
PASSWORD_FILE_NAME = "smtp-password"
YAMNET_TEST_MODE = "yamnet-test"

_COPIED_NOTIFICATION_KEYS = (
    "device_name",
    "timezone",
    "enabled",
    "host",
    "port",
    "sender",
    "recipient",
    "username",
    "starttls",
    "timeout_seconds",
    "status_log",
)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise ValueError(f"unsupported notification setting value: {value!r}")


def _board_notifications(config_path: Path, password: str | None) -> tuple[str, str | None]:
    """Return a board ``[notifications]`` table and the password to bundle."""
    with config_path.open("rb") as file:
        section = tomllib.load(file).get("notifications")
    if not isinstance(section, dict) or section.get("type") != "smtp":
        raise ValueError(f"{config_path} must contain [notifications] with type = 'smtp'")
    settings = {key: section[key] for key in _COPIED_NOTIFICATION_KEYS if key in section}

    settings_file = section.get("settings_file")
    if isinstance(settings_file, str):
        saved = (config_path.parent / settings_file).resolve()
        if saved.is_file():
            from edge_ai.settings import load_notification_preferences

            preferences = load_notification_preferences(saved)
            settings.update(
                recipient=preferences.recipient,
                device_name=preferences.device_name,
                timezone=preferences.timezone,
                enabled=preferences.enabled,
            )

    lines = ["[notifications]", 'type = "smtp"']
    lines += [f"{key} = {_toml_value(value)}" for key, value in settings.items()]
    if "username" not in settings:
        return "\n".join(lines) + "\n", None
    if not password:
        raise ValueError(
            "the SMTP password is required: set the environment variable named by "
            f"[notifications].password_env in {config_path}"
        )
    lines.append(f'password_file = "{PASSWORD_FILE_NAME}"')
    return "\n".join(lines) + "\n", password


def _board_config(config_source: Path, notifications: str | None) -> str:
    source = config_source.read_text(encoding="utf-8")
    if notifications is None:
        return source
    match = re.search(r"^\[notifications\]\s*$", source, flags=re.MULTILINE)
    if match is None:
        return source.rstrip() + "\n\n" + notifications
    if re.search(r"^\[", source[match.end():], flags=re.MULTILINE):
        raise ValueError(f"[notifications] must be the last table in {config_source}")
    return source[: match.start()] + notifications


def _write_synthetic_wavs(destination: Path) -> None:
    """Write deterministic PCM fixtures matching the model-test configuration."""
    import numpy as np

    from edge_ai.inputs.audio import SimulatedSoundInput

    destination.mkdir(parents=True)
    for event in ("background", "smoke_alarm", "glass_break", "fall_thud"):
        frame = SimulatedSoundInput([event], seed=7, loop=False).read()
        pcm = np.where(
            frame.samples < 0.0,
            frame.samples * 32768.0,
            frame.samples * 32767.0,
        ).clip(-32768, 32767).astype("<i2")
        with wave.open(str(destination / f"{event}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(frame.sample_rate)
            output.writeframes(pcm.tobytes())


def package_app(
    destination: Path,
    *,
    mode: str = "demo",
    notifications_config: Path | None = None,
    password: str | None = None,
) -> Path:
    """Write a self-contained app directory and a zip with ``app.yaml`` at its root."""
    if mode not in {"demo", YAMNET_TEST_MODE}:
        raise ValueError(f"unsupported App Lab package mode: {mode!r}")
    if mode == YAMNET_TEST_MODE and notifications_config is not None:
        raise ValueError("the YAMNet model smoke test does not enable notifications")

    notifications: str | None = None
    bundled_password: str | None = None
    if notifications_config is not None:
        notifications, bundled_password = _board_notifications(
            notifications_config, password
        )
    if mode == YAMNET_TEST_MODE:
        name = "private-sound-alerts-yamnet-test"
        config_source = YAMNET_CONFIG_SOURCE
        app_manifest = APP_SOURCE / "app-yamnet-test.yaml"
        requirements = APP_SOURCE / "python" / "requirements-yamnet-test.txt"
    else:
        name = "private-sound-alerts-email" if notifications is not None else "private-sound-alerts"
        config_source = CONFIG_SOURCE
        app_manifest = APP_SOURCE / "app.yaml"
        requirements = APP_SOURCE / "python" / "requirements.txt"

    app_dir = destination / name
    if app_dir.exists():
        shutil.rmtree(app_dir)
    python_dir = app_dir / "python"
    python_dir.mkdir(parents=True)
    shutil.copy2(app_manifest, app_dir / "app.yaml")
    shutil.copytree(APP_SOURCE / "sketch", app_dir / "sketch")
    shutil.copy2(APP_SOURCE / "python" / "main.py", python_dir / "main.py")
    shutil.copy2(requirements, python_dir / "requirements.txt")
    (python_dir / BOARD_CONFIG_NAME).write_text(
        _board_config(config_source, notifications), encoding="utf-8"
    )
    if mode == YAMNET_TEST_MODE:
        if not YAMNET_MODEL_SOURCE.is_file():
            raise ValueError(
                f"YAMNet model not found: {YAMNET_MODEL_SOURCE}; download it as documented"
            )
        model_dir = python_dir / "models"
        model_dir.mkdir()
        shutil.copy2(YAMNET_MODEL_SOURCE, model_dir / "yamnet.onnx")
        shutil.copy2(YAMNET_CLASS_MAP_SOURCE, model_dir / "yamnet_class_map.csv")
        _write_synthetic_wavs(python_dir / "samples")
    if bundled_password is not None:
        secret = python_dir / PASSWORD_FILE_NAME
        secret.write_text(bundled_password + "\n", encoding="utf-8")
        secret.chmod(0o600)
    shutil.copytree(
        PACKAGE_SOURCE,
        python_dir / "edge_ai",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    zip_path = destination / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(app_dir).as_posix())
    return zip_path


def _password_for(config_path: Path) -> str | None:
    with config_path.open("rb") as file:
        section = tomllib.load(file).get("notifications", {})
    name = section.get("password_env") if isinstance(section, dict) else None
    return os.environ.get(name) if isinstance(name, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--notifications",
        type=Path,
        help="private SMTP config whose [notifications] section is bundled for the board",
    )
    parser.add_argument(
        "--mode",
        choices=("demo", YAMNET_TEST_MODE),
        default="demo",
        help="package the default demo or the bundled YAMNet model smoke test",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        zip_path = package_app(
            REPO_ROOT / "dist",
            mode=args.mode,
            notifications_config=args.notifications,
            password=_password_for(args.notifications) if args.notifications else None,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(zip_path)
    if args.notifications is not None:
        print("This zip contains the SMTP password. Import it only on your own board; do not share it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
