"""Explicit TOML configuration factories for the starter pipeline."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import tomllib
from typing import Any

from edge_ai.decision import Decision, decide
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.mock import MockHardware
from edge_ai.hardware.uno_q import UnoQHardware
from edge_ai.inference.base import InferenceEngine, InferenceResult
from edge_ai.inference.dummy import DummyInferenceEngine
from edge_ai.inference.onnx import ONNXInferenceEngine
from edge_ai.inputs.base import InputSource
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput
from edge_ai.inputs.webcam import WebcamInput
from edge_ai.pipeline import Pipeline
from edge_ai.preprocessing.image import preprocess_image
from edge_ai.preprocessing.sensor import normalize_sensor


class ConfigError(ValueError):
    """Raised when a runner configuration is missing or invalid."""


@dataclass(frozen=True)
class ConfiguredPipeline:
    pipeline: Pipeline
    interval_seconds: float


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


def _build_input(section: Mapping[str, Any]) -> InputSource:
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
    raise ConfigError(f"unsupported [inference].type: {component_type!r}")


def _build_decision(section: Mapping[str, Any]) -> Callable[[InferenceResult], Decision]:
    component_type = _component_type(section, "decision")
    if component_type == "default":
        return decide
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


def load_config(path: Path) -> ConfiguredPipeline:
    """Load and validate a configured pipeline from ``path``."""
    config_path, document = _load_document(path)

    runtime = _table(document, "runtime")
    interval_seconds = _number(runtime, "interval_seconds", 0.0)
    if interval_seconds < 0.0:
        raise ConfigError("[runtime].interval_seconds must be non-negative")

    input_source = _build_input(_table(document, "input"))
    try:
        pipeline = Pipeline(
            input_source=input_source,
            preprocessor=_build_preprocessor(_table(document, "preprocessing")),
            inference=_build_inference(_table(document, "inference"), config_path.parent),
            decision_function=_build_decision(_table(document, "decision")),
            hardware=_build_hardware(_table(document, "hardware")),
        )
    except Exception:
        close = getattr(input_source, "close", None)
        if callable(close):
            close()
        raise
    return ConfiguredPipeline(pipeline=pipeline, interval_seconds=interval_seconds)
