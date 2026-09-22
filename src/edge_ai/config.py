"""Explicit TOML configuration factories for the starter pipeline."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import tzinfo
from functools import partial
import os
from pathlib import Path
import tomllib
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from edge_ai.decision import Decision, decide
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.mock import MockHardware
from edge_ai.hardware.uno_q import UnoQHardware
from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.dummy import DummyInferenceEngine
from edge_ai.inference.onnx import ONNXInferenceEngine
from edge_ai.inference.spectral import SpectralSoundInferenceEngine
from edge_ai.inference.yamnet import YAMNetInferenceEngine
from edge_ai.inputs.audio import MicrophoneInput, SimulatedSoundInput, WavAudioInput
from edge_ai.inputs.base import InputSource
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput
from edge_ai.inputs.webcam import WebcamInput
from edge_ai.pipeline import Pipeline
from edge_ai.notifications import AsyncNotifier, Notifier, NotifyingHardware, SMTPNotifier
from edge_ai.preprocessing.audio import extract_audio_features, prepare_audio_waveform
from edge_ai.preprocessing.image import preprocess_image
from edge_ai.preprocessing.sensor import normalize_sensor
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


def _build_input(section: Mapping[str, Any], config_dir: Path) -> InputSource:
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
    if component_type == "microphone":
        device = section.get("device")
        if device is not None and (
            isinstance(device, bool) or not isinstance(device, (str, int))
        ):
            raise ConfigError("[input].device must be a device name or integer index")
        try:
            return MicrophoneInput(
                sample_rate=_integer(section, "sample_rate", 16_000),
                duration_seconds=_number(section, "duration_seconds", 1.0),
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
            )
        except ValueError as exc:
            raise ConfigError(f"invalid [input] configuration: {exc}") from exc
    raise ConfigError(f"unsupported [input].type: {component_type!r}")


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
        return partial(preprocess_image, size=(size[0], size[1]), normalize=normalize)
    if component_type in {"audio_waveform", "audio_features"}:
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
        return partial(
            extract_audio_features,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
    raise ConfigError(f"unsupported [preprocessing].type: {component_type!r}")


def _build_inference(section: Mapping[str, Any], config_dir: Path) -> InferenceEngine:
    component_type = _component_type(section, "inference")
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
        try:
            return YAMNetInferenceEngine(
                (config_dir / model).resolve(),
                background_threshold=_number(section, "background_threshold", 0.1),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
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


def _build_hardware(section: Mapping[str, Any]) -> HardwareBackend:
    component_type = _component_type(section, "hardware")
    if component_type == "mock":
        verbose = section.get("verbose", True)
        if not isinstance(verbose, bool):
            raise ConfigError("[hardware].verbose must be a boolean")
        return MockHardware(verbose=verbose)
    if component_type == "uno_q":
        return UnoQHardware()
    raise ConfigError(f"unsupported [hardware].type: {component_type!r}")


def _build_notifier(
    section: Mapping[str, Any], *, asynchronous: bool = True
) -> ConfiguredNotifier:
    component_type = _component_type(section, "notifications")
    device_name = section.get("device_name", "Home sound monitor")
    if (
        not isinstance(device_name, str)
        or not device_name.strip()
        or "\n" in device_name
        or "\r" in device_name
    ):
        raise ConfigError("[notifications].device_name must be a non-empty string")
    timezone_name = section.get("timezone", "UTC")
    if not isinstance(timezone_name, str) or not timezone_name:
        raise ConfigError("[notifications].timezone must be a non-empty string")
    try:
        local_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(
            f"unknown [notifications].timezone: {timezone_name!r}; use an IANA name"
        ) from exc
    if component_type == "none":
        return ConfiguredNotifier(None, device_name, local_timezone)
    if component_type != "smtp":
        raise ConfigError(f"unsupported [notifications].type: {component_type!r}")

    port = _integer(section, "port", 587)
    if not 1 <= port <= 65_535:
        raise ConfigError("[notifications].port must be between 1 and 65535")
    username = section.get("username")
    if username is not None and (not isinstance(username, str) or not username):
        raise ConfigError("[notifications].username must be a non-empty string")
    password_env = section.get("password_env")
    if password_env is not None and (not isinstance(password_env, str) or not password_env):
        raise ConfigError("[notifications].password_env must be a non-empty string")
    if username is not None and password_env is None:
        raise ConfigError("[notifications].password_env is required when username is set")
    password = os.environ.get(password_env) if password_env is not None else None
    if password_env is not None and password is None:
        raise ConfigError(f"notification secret environment variable is not set: {password_env}")
    timeout_seconds = _number(section, "timeout_seconds", 5.0)
    if timeout_seconds <= 0.0:
        raise ConfigError("[notifications].timeout_seconds must be positive")
    smtp_notifier = SMTPNotifier(
        host=_required_string(section, "host", "notifications"),
        port=port,
        sender=_required_string(section, "sender", "notifications"),
        recipient=_required_string(section, "recipient", "notifications"),
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
    return ConfiguredNotifier(notifier, device_name, local_timezone)


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
    _, document = _load_document(path)
    return _build_notifier(_table(document, "notifications"), asynchronous=asynchronous)


def load_config(path: Path) -> ConfiguredPipeline:
    """Load and validate a configured pipeline from ``path``."""
    config_path, document = _load_document(path)

    runtime = _table(document, "runtime")
    interval_seconds = _number(runtime, "interval_seconds", 0.0)
    if interval_seconds < 0.0:
        raise ConfigError("[runtime].interval_seconds must be non-negative")

    input_source = _build_input(_table(document, "input"), config_path.parent)
    try:
        preprocessor = _build_preprocessor(_table(document, "preprocessing"))
        inference = _build_inference(_table(document, "inference"), config_path.parent)
        decision_function = _build_decision(_table(document, "decision"))
        hardware = _build_hardware(_table(document, "hardware"))
        notifications = document.get("notifications")
        if notifications is not None:
            if not isinstance(notifications, dict):
                raise ConfigError("invalid [notifications] section")
            configured_notifier = _build_notifier(notifications)
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
        )
    except Exception:
        close = getattr(input_source, "close", None)
        if callable(close):
            close()
        raise
    return ConfiguredPipeline(pipeline=pipeline, interval_seconds=interval_seconds)
