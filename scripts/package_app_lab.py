"""Build an Arduino App Lab zip that contains the sound pipeline.

Pass ``--config`` to choose the board pipeline; models referenced anywhere below
its ``[inference]`` section are bundled under ``python/models/``. Pass
``--notifications`` with a private SMTP config to enable email on the board, or
``--mode yamnet-test`` to replay bundled WAV fixtures through YAMNet.
The SMTP password is read from the ignored private config's local ``password`` value,
or from its ``password_env`` variable, and written only to ``python/smtp-password``
inside the git-ignored ``dist/`` output. The literal password is never copied into the
board TOML.
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
YAMNET_MODEL_SOURCE = REPO_ROOT / "models" / "yamnet.onnx"
YAMNET_CLASS_MAP_SOURCE = REPO_ROOT / "models" / "yamnet_class_map.csv"
BOARD_CONFIG_NAME = "sound-uno-q.toml"
PASSWORD_FILE_NAME = "smtp-password"
YAMNET_TEST_MODE = "yamnet-test"
ONNXRUNTIME_REQUIREMENT = "onnxruntime==1.30.0"

YAMNET_TEST_CONFIG = """# Generated UNO Q model smoke test configuration.
[runtime]
interval_seconds = 1.0

[input]
type = "wav"
paths = [
  "samples/background.wav",
  "samples/smoke_alarm.wav",
  "samples/glass_break.wav",
  "samples/fall_thud.wav",
]
loop = true

[preprocessing]
type = "audio_waveform"
sample_rate = 16000
duration_seconds = 1.0
peak_normalize = false

[inference]
type = "yamnet"
model = "models/yamnet.onnx"
background_threshold = 0.10

[decision]
type = "sound_events"
thresholds = { smoke_alarm = 0.25, glass_break = 0.15, fall_thud = 0.20 }
confirmations = 1
hold_seconds = 0.0
notification_cooldown_seconds = 60.0

[hardware]
type = "uno_q"

[notifications]
type = "none"
device_name = "YAMNet model test"
timezone = "America/Los_Angeles"
"""

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
            "the SMTP password is required: set [notifications].password in the "
            f"ignored {config_path}, or set the environment variable named by "
            "[notifications].password_env"
        )
    lines.append(f'password_file = "{PASSWORD_FILE_NAME}"')
    return "\n".join(lines) + "\n", password


def _board_config(source: str, config_path: Path, notifications: str | None) -> str:
    if notifications is None:
        return source
    match = re.search(r"^\[notifications\]\s*$", source, flags=re.MULTILINE)
    if match is None:
        return source.rstrip() + "\n\n" + notifications
    if re.search(r"^\[", source[match.end():], flags=re.MULTILINE):
        raise ValueError(f"[notifications] must be the last table in {config_path}")
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


def _bundled_model(source: str, config_path: Path) -> tuple[str, list[Path], bool]:
    """Point inference model paths at ``python/models/`` and list files to copy.

    Returns the rewritten config, the model files, and whether ONNX Runtime is needed.
    """
    inference = tomllib.loads(source).get("inference", {})
    if not isinstance(inference, dict):
        return source, [], False

    references: list[tuple[str, str | None]] = []

    def visit(section: dict[str, object]) -> None:
        model = section.get("model")
        if isinstance(model, str):
            component_type = section.get("type")
            references.append(
                (model, component_type if isinstance(component_type, str) else None)
            )
        for value in section.values():
            if isinstance(value, dict):
                visit(value)

    visit(inference)
    if not references:
        return source, [], False

    files: list[Path] = []
    names: dict[str, Path] = {}
    replacements: dict[str, str] = {}
    for model, component_type in references:
        model_path = (config_path.parent / model).resolve()
        if not model_path.is_file():
            raise ValueError(
                f"model not found: {model_path}. Download it as described in models/README.md"
            )
        previous = names.get(model_path.name)
        if previous is not None and previous != model_path:
            raise ValueError(f"model filename collision: {previous} and {model_path}")
        names[model_path.name] = model_path
        if model_path not in files:
            files.append(model_path)
        replacements[model] = f"models/{model_path.name}"
        if component_type == "yamnet":
            class_map = model_path.with_name("yamnet_class_map.csv")
            if not class_map.is_file():
                raise ValueError(f"YAMNet class map not found: {class_map}")
            if class_map not in files:
                files.append(class_map)

    pattern = re.compile(r'^(model\s*=\s*)("(?:[^"\\]|\\.)*")', flags=re.MULTILINE)

    def replace_model(match: re.Match[str]) -> str:
        model = json.loads(match.group(2))
        replacement = replacements.get(model)
        if replacement is None:
            return match.group(0)
        return match.group(1) + json.dumps(replacement)

    rewritten, count = pattern.subn(replace_model, source)
    if count < len(references):
        raise ValueError(f"could not rewrite all inference model paths in {config_path}")
    return rewritten, files, True


def package_app(
    destination: Path,
    *,
    mode: str = "demo",
    config_path: Path = CONFIG_SOURCE,
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
        if not YAMNET_MODEL_SOURCE.is_file():
            raise ValueError(
                f"YAMNet model not found: {YAMNET_MODEL_SOURCE}; download it as documented"
            )
        name = "private-sound-alerts-yamnet-test"
        board_config = YAMNET_TEST_CONFIG
        model_files = [YAMNET_MODEL_SOURCE, YAMNET_CLASS_MAP_SOURCE]
        manifest = (APP_SOURCE / "app-yamnet-test.yaml").read_text(encoding="utf-8")
        requirements = (APP_SOURCE / "python" / "requirements-yamnet-test.txt").read_text(
            encoding="utf-8"
        )
    else:
        source, model_files, needs_onnxruntime = _bundled_model(
            config_path.read_text(encoding="utf-8"), config_path
        )
        board_config = _board_config(source, config_path, notifications)

        input_section = tomllib.loads(source).get("input", {})
        input_type = input_section.get("type") if isinstance(input_section, dict) else None
        name = "private-sound-alerts"
        title_suffixes: list[str] = []
        if input_type in {"microphone", "arduino_microphone"}:
            name += "-live"
            title_suffixes.append("live mic")
        elif model_files:
            name += "-yamnet-sim"
            title_suffixes.append("YAMNet, simulated mic")
        if notifications is not None:
            name += "-email"
            title_suffixes.append("email")

        manifest = (APP_SOURCE / "app.yaml").read_text(encoding="utf-8")
        if title_suffixes:
            manifest = re.sub(
                r"^name:\s*(.+)$",
                lambda match: f"name: {match.group(1)} ({', '.join(title_suffixes)})",
                manifest,
                count=1,
                flags=re.MULTILINE,
            )
        requirements = (APP_SOURCE / "python" / "requirements.txt").read_text(encoding="utf-8")
        if needs_onnxruntime:
            requirements = requirements.rstrip() + f"\n{ONNXRUNTIME_REQUIREMENT}\n"

    app_dir = destination / name
    if app_dir.exists():
        shutil.rmtree(app_dir)
    python_dir = app_dir / "python"
    python_dir.mkdir(parents=True)
    (app_dir / "app.yaml").write_text(manifest, encoding="utf-8")
    shutil.copytree(APP_SOURCE / "sketch", app_dir / "sketch")
    shutil.copy2(APP_SOURCE / "python" / "main.py", python_dir / "main.py")
    (python_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
    (python_dir / BOARD_CONFIG_NAME).write_text(board_config, encoding="utf-8")
    if model_files:
        (python_dir / "models").mkdir()
        for path in model_files:
            shutil.copy2(path, python_dir / "models" / path.name)
    if mode == YAMNET_TEST_MODE:
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
    if not isinstance(section, dict):
        return None
    password = section.get("password")
    if password is not None:
        if not isinstance(password, str) or not password:
            raise ValueError("[notifications].password must be a non-empty string")
        return password
    name = section.get("password_env")
    return os.environ.get(name) if isinstance(name, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_SOURCE,
        help="board pipeline config (default: configs/sound-uno-q.toml)",
    )
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
            config_path=args.config.resolve(),
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
