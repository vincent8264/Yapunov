from pathlib import Path

import pytest

from edge_ai.config import ConfigError, load_config, load_hardware_check_config
from edge_ai.hardware.mock import MockHardware
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput


def test_demo_config_builds_pipeline() -> None:
    configured = load_config(Path("configs/demo.toml"))

    assert configured.interval_seconds == 0.5
    assert isinstance(configured.pipeline.input_source, SimulatedSensorInput)
    assert isinstance(configured.pipeline.hardware, MockHardware)


def test_missing_section_has_specific_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[runtime]\ninterval_seconds = 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"\[input\]"):
        load_config(path)


def test_unsupported_component_has_specific_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[runtime]
interval_seconds = 0
[input]
type = "unknown"
[preprocessing]
type = "identity"
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unsupported.*input"):
        load_config(path)


def test_invalid_actuator_range_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hardware.toml"
    path.write_text(
        """
[hardware]
type = "mock"
[hardware_check]
[[hardware_check.actuators]]
type = "pwm"
channel = 0
value = 2.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="value must be 0 to 1"):
        load_hardware_check_config(path)


def test_onnx_requires_confirmed_output_type(tmp_path: Path) -> None:
    path = tmp_path / "onnx.toml"
    path.write_text(
        """
[runtime]
[input]
type = "simulated_sensor"
[preprocessing]
type = "identity"
[inference]
type = "onnx"
model = "model.onnx"
[decision]
type = "default"
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="output_type"):
        load_config(path)
