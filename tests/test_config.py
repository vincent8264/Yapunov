from pathlib import Path
import tomllib

import pytest

from edge_ai.config import (
    ConfigError,
    load_config,
    load_hardware_check_config,
    load_notification_config,
    load_setup_server_config,
)
from edge_ai.hardware.mock import MockHardware
from edge_ai.inputs.audio import MicrophoneHealthInput
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput
from edge_ai.notifications import NotifyingHardware, SMTPNotifier
from edge_ai.settings import NotificationPreferences, save_notification_preferences


def test_demo_config_builds_pipeline() -> None:
    configured = load_config(Path("configs/demo.toml"))

    assert configured.interval_seconds == 0.5
    assert isinstance(configured.pipeline.input_source, SimulatedSensorInput)
    assert isinstance(configured.pipeline.hardware, MockHardware)


def test_arduino_microphone_input_requires_app_lab_runtime(tmp_path: Path) -> None:
    path = tmp_path / "board-live.toml"
    path.write_text(
        """
[runtime]
[input]
type = "arduino_microphone"
[preprocessing]
type = "audio_waveform"
[inference]
type = "spectral_demo"
[decision]
type = "default"
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="App Lab runtime"):
        load_config(path)


def test_arduino_microphone_rejects_invalid_device(tmp_path: Path) -> None:
    path = tmp_path / "board-live.toml"
    path.write_text(
        """
[runtime]
[input]
type = "arduino_microphone"
device = true
[preprocessing]
type = "audio_waveform"
[inference]
type = "spectral_demo"
[decision]
type = "default"
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="device must be"):
        load_config(path)


def test_health_enabled_microphone_open_failure_becomes_runtime_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "health-live.toml"
    path.write_text(
        """
[runtime]
[input]
type = "microphone"
[input.health]
enabled = true
[preprocessing]
type = "audio_waveform"
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    def unavailable_microphone(**_: object) -> object:
        raise RuntimeError("microphone is already in use")

    monkeypatch.setattr("edge_ai.config.MicrophoneInput", unavailable_microphone)

    configured = load_config(path)

    assert isinstance(configured.pipeline.input_source, MicrophoneHealthInput)
    assert configured.pipeline.input_source.source is None
    assert configured.pipeline.step() is None
    assert configured.pipeline.hardware.current_alert == "microphone_fault"


def test_uno_q_sound_config_is_live_yamnet() -> None:
    with Path("configs/sound-uno-q.toml").open("rb") as file:
        config = tomllib.load(file)

    assert config["input"]["type"] == "arduino_microphone"
    assert config["inference"]["type"] == "yamnet"
    assert config["hardware"]["type"] == "uno_q"


def test_live_display_uses_twenty_hz_microphone_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "display.toml"
    path.write_text(
        """
[runtime]
[input]
type = "microphone"
sample_rate = 48000
[preprocessing]
type = "audio_waveform"
duration_seconds = 1.0
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
[display]
type = "audio_spectrum"
rate_hz = 20
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_microphone(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("edge_ai.config.MicrophoneInput", fake_microphone)
    configured = load_config(path)

    assert configured.pipeline.audio_spectrum is not None
    assert configured.pipeline.audio_spectrum.rate_hz == 20
    assert captured["duration_seconds"] == 0.05


def test_board_microphone_display_uses_twenty_hz_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path("configs/sound-uno-q.toml").read_text(encoding="utf-8")
    path = tmp_path / "board-display.toml"
    path.write_text(
        source.replace('model = "../models/yamnet.onnx"\n', "")
        .replace('type = "yamnet"', 'type = "dummy"')
        .replace('type = "uno_q"', 'type = "mock"'),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_microphone(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("edge_ai.config.ArduinoMicrophoneInput", fake_microphone)
    configured = load_config(path)

    assert configured.pipeline.audio_spectrum is not None
    assert configured.pipeline.audio_spectrum.inference_hop_seconds == 0.2
    assert captured["duration_seconds"] == 0.05
    assert isinstance(configured.pipeline.input_source, MicrophoneHealthInput)
    assert configured.pipeline.input_source.source_factory is not None
    assert configured.pipeline.input_source.reopen_on_fault is False


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("silence_threshold", "1.1"),
        ("failure_seconds", "0.0"),
        ("retry_interval_seconds", "0.0"),
    ],
)
def test_microphone_health_rejects_invalid_ranges_before_opening_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
) -> None:
    path = tmp_path / "health.toml"
    path.write_text(
        """
[runtime]
[input]
type = "microphone"
[input.health]
enabled = true
{setting} = {value}
[preprocessing]
type = "audio_waveform"
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
""".format(setting=setting, value=value),
        encoding="utf-8",
    )
    opened = False

    def fake_microphone(**_: object) -> object:
        nonlocal opened
        opened = True
        return object()

    monkeypatch.setattr("edge_ai.config.MicrophoneInput", fake_microphone)

    with pytest.raises(ConfigError, match=setting):
        load_config(path)
    assert opened is False


def test_live_display_accepts_a_shorter_inference_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "display.toml"
    path.write_text(
        """
[runtime]
[input]
type = "microphone"
[preprocessing]
type = "audio_waveform"
duration_seconds = 1.0
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
[display]
type = "audio_spectrum"
rate_hz = 20
inference_hop_seconds = 0.2
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("edge_ai.config.MicrophoneInput", lambda **_: object())

    configured = load_config(path)

    assert configured.pipeline.audio_spectrum is not None
    assert configured.pipeline.audio_spectrum.inference_hop_seconds == 0.2


def test_display_rejects_non_microphone_input(tmp_path: Path) -> None:
    path = tmp_path / "sim-display.toml"
    path.write_text(
        """
[runtime]
[input]
type = "simulated_sound"
[preprocessing]
type = "audio_features"
[inference]
type = "spectral_demo"
[decision]
type = "default"
[hardware]
type = "mock"
[display]
type = "audio_spectrum"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="arduino_microphone"):
        load_config(path)


def test_sound_decision_rejects_invalid_threshold(tmp_path: Path) -> None:
    path = tmp_path / "bad-sound.toml"
    path.write_text(
        """
[runtime]
[input]
type = "simulated_sound"
events = ["smoke_alarm"]
[preprocessing]
type = "audio_features"
[inference]
type = "spectral_demo"
[decision]
type = "sound_events"
thresholds = { smoke_alarm = 2.0 }
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="thresholds must be between"):
        load_config(path)


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


def test_yamnet_reports_missing_model_at_startup(tmp_path: Path) -> None:
    path = tmp_path / "yamnet.toml"
    path.write_text(
        """
[runtime]
[input]
type = "simulated_sound"
events = ["background"]
[preprocessing]
type = "audio_waveform"
[inference]
type = "yamnet"
model = "missing.onnx"
[decision]
type = "sound_events"
thresholds = { smoke_alarm = 0.25, glass_break = 0.2, fall_thud = 0.2 }
confirmations = { smoke_alarm = 2, glass_break = 1, fall_thud = 1 }
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="YAMNet ONNX model not found"):
        load_config(path)


def test_scheduled_audio_requires_both_detector_tables(tmp_path: Path) -> None:
    path = tmp_path / "scheduled.toml"
    path.write_text(
        """
[runtime]
[input]
type = "simulated_sound"
events = ["background"]
[preprocessing]
type = "audio_sliding_window"
[inference]
type = "scheduled_audio"
[decision]
type = "sound_events"
thresholds = { help_call = 0.5 }
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"\[inference.environment\].*tables"):
        load_config(path)


def test_keyword_spotter_reports_missing_model_at_startup(tmp_path: Path) -> None:
    path = tmp_path / "keyword.toml"
    path.write_text(
        """
[runtime]
[input]
type = "simulated_sound"
events = ["background"]
[preprocessing]
type = "audio_waveform"
[inference]
type = "keyword_spotter"
model = "missing.onnx"
[decision]
type = "sound_events"
thresholds = { help_call = 0.5 }
[hardware]
type = "mock"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="keyword ONNX model not found"):
        load_config(path)


def test_notification_config_loads_without_pipeline_sections(tmp_path: Path) -> None:
    path = tmp_path / "notifications.toml"
    path.write_text(
        """
[notifications]
type = "smtp"
device_name = "Living room"
timezone = "America/Los_Angeles"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
starttls = true
status_log = true
""",
        encoding="utf-8",
    )

    configured = load_notification_config(path)

    assert configured.notifier is not None
    assert configured.notifier.channel == "email"
    assert configured.device_name == "Living room"
    assert str(configured.local_timezone) == "America/Los_Angeles"


def test_setup_server_config_resolves_private_files_from_config(tmp_path: Path) -> None:
    path = tmp_path / "board.toml"
    path.write_text(
        """
[setup]
enabled = true
host = "0.0.0.0"
port = 8123
pin_file = "private/setup-pin"
""",
        encoding="utf-8",
    )

    configured = load_setup_server_config(path)

    assert configured.enabled
    assert configured.host == "0.0.0.0"
    assert configured.port == 8123
    assert configured.pin_path == (tmp_path / "private" / "setup-pin").resolve()


def test_lan_setup_server_requires_a_pin_file(tmp_path: Path) -> None:
    path = tmp_path / "board.toml"
    path.write_text('[setup]\nenabled = true\nhost = "0.0.0.0"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="pin_file.*required"):
        load_setup_server_config(path)


def test_unknown_notification_timezone_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notifications.toml"
    path.write_text(
        """
[notifications]
type = "none"
timezone = "Mars/Olympus_Mons"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown.*timezone"):
        load_notification_config(path)


def test_private_live_example_builds_email_notifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDGE_AI_SMTP_PASSWORD", "test-only-password")

    configured = load_notification_config(Path("configs/private-live.example.toml"))

    assert configured.notifier is not None
    assert configured.notifier.channel == "email"
    assert configured.device_name == "Living room sound monitor"


def test_saved_preferences_override_checked_in_notification_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "private.toml"
    config.write_text(
        """
[notifications]
type = "smtp"
settings_file = "settings.json"
device_name = "Default room"
timezone = "UTC"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "default@example.com"
username = "monitor@example.com"
password_env = "TEST_SMTP_PASSWORD"
""",
        encoding="utf-8",
    )
    save_notification_preferences(
        tmp_path / "settings.json",
        NotificationPreferences(
            "family@example.com", "Kitchen", "America/Los_Angeles", True
        ),
    )
    monkeypatch.setenv("TEST_SMTP_PASSWORD", "secret")

    configured = load_notification_config(config)

    assert isinstance(configured.notifier, SMTPNotifier)
    assert configured.notifier.recipient == "family@example.com"
    assert configured.device_name == "Kitchen"
    assert str(configured.local_timezone) == "America/Los_Angeles"


def test_password_file_supplies_smtp_secret(tmp_path: Path) -> None:
    config = tmp_path / "board.toml"
    config.write_text(
        """
[notifications]
type = "smtp"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
username = "monitor@example.com"
password_file = "smtp-password"
""",
        encoding="utf-8",
    )
    (tmp_path / "smtp-password").write_text("file-secret\n", encoding="utf-8")

    configured = load_notification_config(config)

    assert isinstance(configured.notifier, SMTPNotifier)
    assert configured.notifier.password == "file-secret"


def test_missing_password_file_is_rejected_at_startup(tmp_path: Path) -> None:
    config = tmp_path / "board.toml"
    config.write_text(
        """
[notifications]
type = "smtp"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
username = "monitor@example.com"
password_file = "missing-password"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="password file not found"):
        load_notification_config(config)


def test_password_env_and_file_are_mutually_exclusive(tmp_path: Path) -> None:
    config = tmp_path / "board.toml"
    config.write_text(
        """
[notifications]
type = "smtp"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
username = "monitor@example.com"
password_env = "TEST_SMTP_PASSWORD"
password_file = "smtp-password"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="only one"):
        load_notification_config(config)


def test_disabled_saved_preferences_do_not_require_smtp_secret(tmp_path: Path) -> None:
    config = tmp_path / "private.toml"
    config.write_text(
        """
[notifications]
type = "smtp"
settings_file = "settings.json"
device_name = "Default room"
timezone = "UTC"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "default@example.com"
username = "monitor@example.com"
password_env = "MISSING_TEST_SMTP_PASSWORD"
""",
        encoding="utf-8",
    )
    save_notification_preferences(
        tmp_path / "settings.json",
        NotificationPreferences("family@example.com", "Kitchen", "UTC", False),
    )

    configured = load_notification_config(config)

    assert configured.notifier is None
    assert configured.device_name == "Kitchen"


def test_disabled_smtp_pipeline_can_be_enabled_later_without_restart(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.toml"
    config.write_text(
        """
[runtime]
[input]
type = "simulated_sensor"
[preprocessing]
type = "identity"
[inference]
type = "dummy"
[decision]
type = "default"
[hardware]
type = "mock"
verbose = false
[notifications]
type = "smtp"
settings_file = "settings.json"
host = "smtp.example.com"
sender = "monitor@example.com"
recipient = "family@example.com"
""",
        encoding="utf-8",
    )
    save_notification_preferences(
        tmp_path / "settings.json",
        NotificationPreferences("family@example.com", "Kitchen", "UTC", False),
    )

    configured = load_config(config)

    assert isinstance(configured.pipeline.hardware, NotifyingHardware)
    assert configured.pipeline.hardware.notifier is None
