"""Explicit TOML configuration factories for the starter pipeline."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import tzinfo
from functools import partial
import os
from pathlib import Path
import tomllib
from typing import Any
from zoneinfo import ZoneInfo

from edge_ai.decision import Decision, decide
from edge_ai.audio_display import AudioSpectrum
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.mock import MockHardware
from edge_ai.hardware.matrix_preview import MatrixPreviewHardware
from edge_ai.hardware.uno_q import UnoQHardware
from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.dummy import DummyInferenceEngine
from edge_ai.inference.spectral import SpectralSoundInferenceEngine
from edge_ai.inputs.audio import (
    ArduinoMicrophoneInput,
    MicrophoneInput,
    SimulatedSoundInput,
    WavAudioInput,
)
from edge_ai.inputs.base import InputSource
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput
from edge_ai.pipeline import Pipeline
from edge_ai.notifications import AsyncNotifier, Notifier, NotifyingHardware, SMTPNotifier
from edge_ai.preprocessing.audio import (
    SlidingAudioWindow,
    extract_audio_features,
    prepare_audio_waveform,
)
from edge_ai.preprocessing.sensor import normalize_sensor
from edge_ai.settings import (
    NotificationPreferences,
    SettingsError,
    load_notification_preferences,
)
from edge_ai.risk_monitor import RiskRule, RiskWarningDecision, RollingRiskMonitor
from edge_ai.sound_decision import SoundDecisionPolicy


class ConfigError(ValueError):
    """Raised when a runner configuration is missing or invalid."""


@dataclass(frozen=True)
class ConfiguredPipeline:
    pipeline: Pipeline
    interval_seconds: float


@dataclass(frozen=True)
class ConfiguredNotifier:
    notifier: Notifier | None
    device_name: str
    local_timezone: tzinfo


@dataclass(frozen=True)
class NotificationSetupConfig:
    config_path: Path
    settings_path: Path
    preferences: NotificationPreferences


@dataclass(frozen=True)
class ActuatorCheck:
    kind: str
    channel: int
    value: float
    safe_value: float


@dataclass(frozen=True)
class HardwareCheckPlan:
    hardware: HardwareBackend
    check_led: bool
    led_hold_seconds: float
    actuators: tuple[ActuatorCheck, ...]


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] section")
    return value


def _component_type(section: Mapping[str, Any], section_name: str) -> str:
    value = section.get("type")
    if not isinstance(value, str) or not value:
        raise ConfigError(f"[{section_name}].type must be a non-empty string")
    return value


def _number(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _integer(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _boolean(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _required_string(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"[{section_name}].{key} must be a non-empty string")
    return value


def _build_input(
    section: Mapping[str, Any],
    config_dir: Path,
    *,
    frame_duration_seconds: float | None = None,
) -> InputSource:
    component_type = _component_type(section, "input")
    if component_type == "simulated_sensor":
        seed = section.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ConfigError("[input].seed must be an integer")
        try:
            return SimulatedSensorInput(
                normal_value=_number(section, "normal_value", 0.3),
                noise=_number(section, "noise", 0.03),
                anomaly_probability=_number(section, "anomaly_probability", 0.1),
                anomaly_value=_number(section, "anomaly_value", 1.3),
                seed=seed,
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [input] configuration: {exc}") from exc
    if component_type == "webcam":
        camera_index = section.get("camera_index", 0)
        if isinstance(camera_index, bool) or not isinstance(camera_index, int):
            raise ConfigError("[input].camera_index must be an integer")
        from edge_ai.inputs.webcam import WebcamInput

        return WebcamInput(camera_index=camera_index)
    if component_type == "wav":
        raw_paths = section.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths or not all(
            isinstance(path, str) and path for path in raw_paths
        ):
            raise ConfigError("[input].paths must be a non-empty array of WAV paths")
        try:
            return WavAudioInput(
                [(config_dir / path).resolve() for path in raw_paths],
                loop=_boolean(section, "loop", True),
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [input] configuration: {exc}") from exc
    if component_type in {"microphone", "arduino_microphone"}:
        device = section.get("device")
        if device is not None and (
            isinstance(device, bool) or not isinstance(device, (str, int))
        ):
            raise ConfigError("[input].device must be a device name or integer index")
        microphone_type = (
            MicrophoneInput if component_type == "microphone" else ArduinoMicrophoneInput
        )
        try:
            return microphone_type(
                sample_rate=_integer(section, "sample_rate", 16_000),
                duration_seconds=frame_duration_seconds
                if frame_duration_seconds is not None
                else _number(section, "duration_seconds", 1.0),
                device=device,
            )
        except (RuntimeError, ValueError) as exc:
            raise ConfigError(f"invalid [input] configuration: {exc}") from exc
    if component_type == "simulated_sound":
        events = section.get(
            "events",
            ["background", "smoke_alarm", "background", "glass_break", "background", "fall_thud"],
        )
        if not isinstance(events, list) or not all(isinstance(event, str) for event in events):
            raise ConfigError("[input].events must be an array of strings")
        try:
            return SimulatedSoundInput(
                events,
                sample_rate=_integer(section, "sample_rate", 16_000),
                duration_seconds=_number(section, "duration_seconds", 1.0),
                seed=_integer(section, "seed", 7),
                loop=_boolean(section, "loop", True),
                choose_randomly=_boolean(section, "random", False),
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [input] configuration: {exc}") from exc
    raise ConfigError(f"unsupported [input].type: {component_type!r}")


def _build_audio_spectrum(
    document: Mapping[str, Any],
    *,
    input_section: Mapping[str, Any],
    preprocessing_section: Mapping[str, Any],
) -> AudioSpectrum | None:
    section = document.get("display")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError("invalid [display] section")
    if _component_type(section, "display") != "audio_spectrum":
        raise ConfigError(f"unsupported [display].type: {section.get('type')!r}")
    if _component_type(input_section, "input") not in {
        "microphone",
        "arduino_microphone",
    }:
        raise ConfigError(
            "[display] audio_spectrum requires [input].type = 'microphone' "
            "or 'arduino_microphone'"
        )
    if _component_type(preprocessing_section, "preprocessing") not in {
        "audio_waveform",
        "audio_features",
    }:
        raise ConfigError("[display] audio_spectrum requires audio preprocessing")
    try:
        return AudioSpectrum(
            rate_hz=_integer(section, "rate_hz", 20),
            inference_duration_seconds=_number(
                preprocessing_section, "duration_seconds", 1.0
            ),
            inference_hop_seconds=(
                _number(section, "inference_hop_seconds", 1.0)
                if "inference_hop_seconds" in section
                else None
            ),
            floor_db=_number(section, "floor_db", -60.0),
            ceiling_db=_number(section, "ceiling_db", -6.0),
        )
    except ValueError as exc:
        raise ConfigError(f"invalid [display] configuration: {exc}") from exc


def _identity(value: Any) -> Any:
    return value


def _build_preprocessor(section: Mapping[str, Any]) -> Callable[[Any], Any]:
    component_type = _component_type(section, "preprocessing")
    if component_type == "identity":
        return _identity
    if component_type == "normalize_sensor":
        scale = _number(section, "scale", 1.0)
        if scale == 0.0:
            raise ConfigError("[preprocessing].scale must not be zero")
        return partial(
            normalize_sensor,
            mean=_number(section, "mean", 0.0),
            scale=scale,
        )
    if component_type == "image":
        size = section.get("size", [224, 224])
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in size)
        ):
            raise ConfigError("[preprocessing].size must contain two positive integers")
        normalize = section.get("normalize", True)
        if not isinstance(normalize, bool):
            raise ConfigError("[preprocessing].normalize must be a boolean")
        from edge_ai.preprocessing.image import preprocess_image

        return partial(preprocess_image, size=(size[0], size[1]), normalize=normalize)
    if component_type in {"audio_waveform", "audio_features", "audio_sliding_window"}:
        sample_rate = _integer(section, "sample_rate", 16_000)
        duration_seconds = _number(section, "duration_seconds", 1.0)
        if sample_rate < 1:
            raise ConfigError("[preprocessing].sample_rate must be positive")
        if duration_seconds <= 0.0:
            raise ConfigError("[preprocessing].duration_seconds must be positive")
        if component_type == "audio_waveform":
            return partial(
                prepare_audio_waveform,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                peak_normalize=_boolean(section, "peak_normalize", False),
            )
        if component_type == "audio_sliding_window":
            return SlidingAudioWindow(
                sample_rate=sample_rate,
                window_seconds=duration_seconds,
                peak_normalize=_boolean(section, "peak_normalize", False),
            )
        return partial(
            extract_audio_features,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
    raise ConfigError(f"unsupported [preprocessing].type: {component_type!r}")


def _build_inference(
    section: Mapping[str, Any],
    config_dir: Path,
    *,
    watched_labels: tuple[str, ...] = (),
) -> InferenceEngine:
    component_type = _component_type(section, "inference")
    if watched_labels and component_type not in {"yamnet", "scheduled_audio"}:
        raise ConfigError(
            f"[decision.risks] requires YAMNet inference, not [inference].type = "
            f"{component_type!r}"
        )
    if component_type == "dummy":
        try:
            return DummyInferenceEngine(threshold=_number(section, "threshold", 0.8))
        except ValueError as exc:
            raise ConfigError(f"invalid [inference] configuration: {exc}") from exc
    if component_type == "onnx":
        model = section.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigError("[inference].model must be a non-empty path")
        labels = section.get("labels")
        if labels is not None and (
            not isinstance(labels, list) or not all(isinstance(label, str) for label in labels)
        ):
            raise ConfigError("[inference].labels must be an array of strings")
        output_type = section.get("output_type")
        if output_type != "probabilities":
            raise ConfigError(
                "[inference].output_type must be 'probabilities' after confirming the "
                "model output activation; otherwise add a model-specific output adapter"
            )
        model_path = (config_dir / model).resolve()
        from edge_ai.inference.onnx import ONNXInferenceEngine

        try:
            return ONNXInferenceEngine(
                model_path,
                labels=labels,
                output_type=output_type,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    if component_type == "spectral_demo":
        try:
            return SpectralSoundInferenceEngine(min_rms=_number(section, "min_rms", 0.02))
        except ValueError as exc:
            raise ConfigError(f"invalid [inference] configuration: {exc}") from exc
    if component_type == "yamnet":
        model = section.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigError("[inference].model must be a non-empty path")
        from edge_ai.inference.yamnet import YAMNetInferenceEngine

        try:
            return YAMNetInferenceEngine(
                (config_dir / model).resolve(),
                background_threshold=_number(section, "background_threshold", 0.1),
                watched_labels=watched_labels,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    if component_type == "keyword_spotter":
        model = section.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigError("[inference].model must be a non-empty path")
        from edge_ai.inference.keyword import KeywordSpotterInferenceEngine

        try:
            return KeywordSpotterInferenceEngine(
                (config_dir / model).resolve(),
                activation_threshold=_number(section, "activation_threshold", 0.5),
                positive_index=_integer(section, "positive_index", 1),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    if component_type == "scheduled_audio":
        environment = section.get("environment")
        keyword = section.get("keyword")
        if not isinstance(environment, dict) or not isinstance(keyword, dict):
            raise ConfigError(
                "[inference.environment] and [inference.keyword] must be tables"
            )
        from edge_ai.inference.scheduled import ScheduledAudioInferenceEngine

        try:
            return ScheduledAudioInferenceEngine(
                _build_inference(environment, config_dir, watched_labels=watched_labels),
                _build_inference(keyword, config_dir),
                environment_every_steps=_integer(
                    section, "environment_every_steps", 3
                ),
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [inference] configuration: {exc}") from exc
    raise ConfigError(f"unsupported [inference].type: {component_type!r}")


def _build_decision(section: Mapping[str, Any]) -> Callable[[InferenceResult], Decision]:
    component_type = _component_type(section, "decision")
    if component_type == "default":
        return decide
    if component_type == "sound_events":
        raw_thresholds = section.get("thresholds")
        if not isinstance(raw_thresholds, dict) or not raw_thresholds:
            raise ConfigError("[decision].thresholds must be a non-empty table")
        thresholds: dict[str, float] = {}
        for label, value in raw_thresholds.items():
            if (
                not isinstance(label, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ConfigError("[decision].thresholds must map labels to numbers")
            thresholds[label] = float(value)
        raw_confirmations = section.get("confirmations", 2)
        confirmations: int | dict[str, int]
        if isinstance(raw_confirmations, bool):
            raise ConfigError("[decision].confirmations must be an integer or table")
        if isinstance(raw_confirmations, int):
            confirmations = raw_confirmations
        elif isinstance(raw_confirmations, dict):
            if not all(
                isinstance(label, str)
                and not isinstance(value, bool)
                and isinstance(value, int)
                for label, value in raw_confirmations.items()
            ):
                raise ConfigError("[decision].confirmations must map labels to integers")
            confirmations = dict(raw_confirmations)
        else:
            raise ConfigError("[decision].confirmations must be an integer or table")
        try:
            return SoundDecisionPolicy(
                thresholds,
                confirmations=confirmations,
                hold_seconds=_number(section, "hold_seconds", 3.0),
                notification_cooldown_seconds=_number(
                    section, "notification_cooldown_seconds", 60.0
                ),
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [decision] configuration: {exc}") from exc
    raise ConfigError(f"unsupported [decision].type: {component_type!r}")


def _build_risk_rules(section: Mapping[str, Any]) -> tuple[RiskRule, ...]:
    raw_risks = section.get("risks")
    if raw_risks is None:
        return ()
    if not isinstance(raw_risks, dict) or not raw_risks:
        raise ConfigError("[decision.risks] must contain at least one [decision.risks.<name>]")
    rules: list[RiskRule] = []
    for name, risk in raw_risks.items():
        table = f"decision.risks.{name}"
        if not isinstance(risk, dict):
            raise ConfigError(f"[{table}] must be a table")
        labels = risk.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
        ):
            raise ConfigError(f"[{table}].labels must be a non-empty array of YAMNet labels")
        if "threshold" not in risk:
            raise ConfigError(f"[{table}].threshold is required")
        try:
            rules.append(
                RiskRule(
                    name=name,
                    message=_required_string(risk, "message", table),
                    labels=tuple(labels),
                    threshold=_number(risk, "threshold", 0.0),
                    window_seconds=_number(risk, "window_seconds", 15.0),
                    min_detected_seconds=_number(risk, "min_detected_seconds", 7.0),
                    cooldown_seconds=_number(risk, "cooldown_seconds", 300.0),
                ).validated()
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [{table}] configuration: {exc}") from exc
    return tuple(rules)


def _build_hardware(section: Mapping[str, Any]) -> HardwareBackend:
    component_type = _component_type(section, "hardware")
    if component_type == "mock":
        verbose = section.get("verbose", True)
        if not isinstance(verbose, bool):
            raise ConfigError("[hardware].verbose must be a boolean")
        return MockHardware(verbose=verbose)
    if component_type == "uno_q":
        return UnoQHardware()
    if component_type == "matrix_preview":
        pixel_size = _integer(section, "pixel_size", 28)
        try:
            return MatrixPreviewHardware(pixel_size=pixel_size)
        except (RuntimeError, ValueError) as exc:
            raise ConfigError(f"invalid [hardware] matrix preview configuration: {exc}") from exc
    raise ConfigError(f"unsupported [hardware].type: {component_type!r}")


def _notification_settings_path(
    section: Mapping[str, Any], config_dir: Path, *, required: bool = False
) -> Path | None:
    value = section.get("settings_file")
    if value is None:
        if required:
            raise ConfigError("[notifications].settings_file is required for the setup portal")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("[notifications].settings_file must be a non-empty path")
    return (config_dir / value).resolve()


def _default_notification_preferences(
    section: Mapping[str, Any], *, require_recipient: bool
) -> NotificationPreferences:
    device_name = section.get("device_name", "Home sound monitor")
    timezone_name = section.get("timezone", "UTC")
    recipient = section.get("recipient")
    if recipient is None and not require_recipient:
        recipient = "disabled@example.invalid"
    enabled = section.get("enabled", True)
    if not isinstance(device_name, str):
        raise ConfigError("[notifications].device_name must be a string")
    if not isinstance(timezone_name, str):
        raise ConfigError("[notifications].timezone must be a string")
    if not isinstance(recipient, str):
        raise ConfigError("[notifications].recipient must be a non-empty string")
    if not isinstance(enabled, bool):
        raise ConfigError("[notifications].enabled must be a boolean")
    try:
        return NotificationPreferences(recipient, device_name, timezone_name, enabled).validated()
    except SettingsError as exc:
        raise ConfigError(f"invalid [notifications] preferences: {exc}") from exc


def _effective_notification_preferences(
    section: Mapping[str, Any], config_dir: Path, *, require_recipient: bool
) -> NotificationPreferences:
    defaults = _default_notification_preferences(section, require_recipient=require_recipient)
    settings_path = _notification_settings_path(section, config_dir)
    if settings_path is None or not settings_path.exists():
        return defaults
    try:
        return load_notification_preferences(settings_path)
    except SettingsError as exc:
        raise ConfigError(str(exc)) from exc


def _build_notifier(
    section: Mapping[str, Any],
    config_dir: Path,
    *,
    asynchronous: bool = True,
    preferences_override: NotificationPreferences | None = None,
    force_enabled: bool = False,
) -> ConfiguredNotifier:
    component_type = _component_type(section, "notifications")
    preferences = preferences_override or _effective_notification_preferences(
        section, config_dir, require_recipient=component_type != "none"
    )
    try:
        preferences = preferences.validated()
    except SettingsError as exc:
        raise ConfigError(f"invalid notification preferences: {exc}") from exc
    local_timezone = ZoneInfo(preferences.timezone)
    if component_type == "none":
        return ConfiguredNotifier(None, preferences.device_name, local_timezone)
    if component_type != "smtp":
        raise ConfigError(f"unsupported [notifications].type: {component_type!r}")
    if not preferences.enabled and not force_enabled:
        return ConfiguredNotifier(None, preferences.device_name, local_timezone)

    port = _integer(section, "port", 587)
    if not 1 <= port <= 65_535:
        raise ConfigError("[notifications].port must be between 1 and 65535")
    username = section.get("username")
    if username is not None and (not isinstance(username, str) or not username):
        raise ConfigError("[notifications].username must be a non-empty string")
    password_env = section.get("password_env")
    if password_env is not None and (not isinstance(password_env, str) or not password_env):
        raise ConfigError("[notifications].password_env must be a non-empty string")
    password_file = section.get("password_file")
    if password_file is not None and (not isinstance(password_file, str) or not password_file):
        raise ConfigError("[notifications].password_file must be a non-empty path")
    if password_env is not None and password_file is not None:
        raise ConfigError("set only one of [notifications].password_env or password_file")
    if username is not None and password_env is None and password_file is None:
        raise ConfigError(
            "[notifications].password_env or password_file is required when username is set"
        )
    password: str | None = None
    if password_env is not None:
        password = os.environ.get(password_env)
        if password is None:
            raise ConfigError(
                f"notification secret environment variable is not set: {password_env}"
            )
    if password_file is not None:
        secret_path = (config_dir / password_file).resolve()
        try:
            password = secret_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise ConfigError(f"notification password file not found: {secret_path}") from exc
        except OSError as exc:
            raise ConfigError(f"could not read notification password file: {exc}") from exc
        if not password:
            raise ConfigError(f"notification password file is empty: {secret_path}")
    timeout_seconds = _number(section, "timeout_seconds", 5.0)
    if timeout_seconds <= 0.0:
        raise ConfigError("[notifications].timeout_seconds must be positive")
    smtp_notifier = SMTPNotifier(
        host=_required_string(section, "host", "notifications"),
        port=port,
        sender=_required_string(section, "sender", "notifications"),
        recipient=preferences.recipient,
        username=username,
        password=password,
        starttls=_boolean(section, "starttls", True),
        timeout_seconds=timeout_seconds,
    )
    status_log = _boolean(section, "status_log", True)
    notifier: Notifier = smtp_notifier
    if asynchronous:
        notifier = AsyncNotifier(
            smtp_notifier,
            reporter=print if status_log else None,
            close_timeout_seconds=timeout_seconds + 1.0,
        )
    return ConfiguredNotifier(notifier, preferences.device_name, local_timezone)


def _load_document(path: Path) -> tuple[Path, Mapping[str, Any]]:
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as file:
            return config_path, tomllib.load(file)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc


def load_hardware_check_config(path: Path) -> HardwareCheckPlan:
    """Load safe hardware diagnostics from a TOML file."""
    _, document = _load_document(path)
    section = _table(document, "hardware_check")
    check_led = section.get("led", True)
    if not isinstance(check_led, bool):
        raise ConfigError("[hardware_check].led must be a boolean")
    led_hold_seconds = _number(section, "led_hold_seconds", 0.25)
    if not 0.0 <= led_hold_seconds <= 5.0:
        raise ConfigError("[hardware_check].led_hold_seconds must be between 0 and 5")

    raw_actuators = section.get("actuators", [])
    if not isinstance(raw_actuators, list):
        raise ConfigError("[[hardware_check.actuators]] must be an array of tables")
    actuators: list[ActuatorCheck] = []
    for index, item in enumerate(raw_actuators):
        if not isinstance(item, dict):
            raise ConfigError(f"hardware_check actuator {index} must be a table")
        kind = item.get("type")
        channel = item.get("channel")
        value = item.get("value")
        safe_value = item.get("safe_value", 0.0 if kind == "pwm" else None)
        if kind not in {"pwm", "servo"}:
            raise ConfigError(f"hardware_check actuator {index} has unsupported type {kind!r}")
        if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
            raise ConfigError(f"hardware_check actuator {index} channel must be non-negative")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"hardware_check actuator {index} value must be a number")
        if isinstance(safe_value, bool) or not isinstance(safe_value, (int, float)):
            raise ConfigError(
                f"hardware_check actuator {index} safe_value must be specified as a number"
            )
        numeric_value = float(value)
        numeric_safe_value = float(safe_value)
        if kind == "pwm" and (
            not 0.0 <= numeric_value <= 1.0 or not 0.0 <= numeric_safe_value <= 1.0
        ):
            raise ConfigError(f"hardware_check PWM actuator {index} value must be 0 to 1")
        if kind == "servo" and (
            not 0.0 <= numeric_value <= 180.0
            or not 0.0 <= numeric_safe_value <= 180.0
        ):
            raise ConfigError(f"hardware_check servo actuator {index} value must be 0 to 180")
        actuators.append(ActuatorCheck(kind, channel, numeric_value, numeric_safe_value))

    return HardwareCheckPlan(
        hardware=_build_hardware(_table(document, "hardware")),
        check_led=check_led,
        led_hold_seconds=led_hold_seconds,
        actuators=tuple(actuators),
    )


def load_notification_config(path: Path, *, asynchronous: bool = False) -> ConfiguredNotifier:
    """Load only notification settings, without constructing the AI pipeline."""
    config_path, document = _load_document(path)
    return _build_notifier(
        _table(document, "notifications"),
        config_path.parent,
        asynchronous=asynchronous,
    )


def load_notification_setup_config(path: Path) -> NotificationSetupConfig:
    """Load editable preferences without constructing the model or requiring SMTP secrets."""
    config_path, document = _load_document(path)
    section = _table(document, "notifications")
    if _component_type(section, "notifications") != "smtp":
        raise ConfigError("the setup portal requires [notifications].type = 'smtp'")
    settings_path = _notification_settings_path(section, config_path.parent, required=True)
    assert settings_path is not None
    preferences = _effective_notification_preferences(
        section, config_path.parent, require_recipient=True
    )
    return NotificationSetupConfig(config_path, settings_path, preferences)


def build_notification_test_notifier(
    path: Path, preferences: NotificationPreferences
) -> ConfiguredNotifier:
    """Build a synchronous notifier for testing unsaved portal preferences."""
    config_path, document = _load_document(path)
    return _build_notifier(
        _table(document, "notifications"),
        config_path.parent,
        asynchronous=False,
        preferences_override=preferences,
        force_enabled=True,
    )


def load_config(path: Path) -> ConfiguredPipeline:
    """Load and validate a configured pipeline from ``path``."""
    config_path, document = _load_document(path)

    runtime = _table(document, "runtime")
    interval_seconds = _number(runtime, "interval_seconds", 0.0)
    if interval_seconds < 0.0:
        raise ConfigError("[runtime].interval_seconds must be non-negative")

    input_section = _table(document, "input")
    preprocessing_section = _table(document, "preprocessing")
    audio_spectrum = _build_audio_spectrum(
        document, input_section=input_section, preprocessing_section=preprocessing_section
    )
    input_source = _build_input(
        input_section,
        config_path.parent,
        frame_duration_seconds=audio_spectrum.frame_duration_seconds if audio_spectrum else None,
    )
    try:
        preprocessor = _build_preprocessor(preprocessing_section)
        decision_section = _table(document, "decision")
        risk_rules = _build_risk_rules(decision_section)
        inference = _build_inference(
            _table(document, "inference"),
            config_path.parent,
            watched_labels=tuple(
                dict.fromkeys(label for rule in risk_rules for label in rule.labels)
            ),
        )
        decision_function = _build_decision(decision_section)
        if risk_rules:
            decision_function = RiskWarningDecision(
                decision_function,
                RollingRiskMonitor(
                    risk_rules,
                    max_observation_seconds=_number(
                        preprocessing_section, "duration_seconds", 1.0
                    ),
                ),
            )
        hardware = _build_hardware(_table(document, "hardware"))
        notifications = document.get("notifications")
        if notifications is not None:
            if not isinstance(notifications, dict):
                raise ConfigError("invalid [notifications] section")
            configured_notifier = _build_notifier(notifications, config_path.parent)
            if configured_notifier.notifier is not None:
                hardware = NotifyingHardware(
                    hardware,
                    configured_notifier.notifier,
                    device_name=configured_notifier.device_name,
                    local_timezone=configured_notifier.local_timezone,
                )
        pipeline = Pipeline(
            input_source=input_source,
            preprocessor=preprocessor,
            inference=inference,
            decision_function=decision_function,
            hardware=hardware,
            audio_spectrum=audio_spectrum,
        )
    except Exception:
        close = getattr(input_source, "close", None)
        if callable(close):
            close()
        raise
    return ConfiguredPipeline(pipeline=pipeline, interval_seconds=interval_seconds)
