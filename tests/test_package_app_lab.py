from pathlib import Path
import importlib.util
import tomllib
import zipfile

import pytest

from edge_ai.config import load_notification_config

_SPEC = importlib.util.spec_from_file_location(
    "package_app_lab",
    Path(__file__).parents[1] / "scripts" / "package_app_lab.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_PRIVATE_CONFIG = """
[notifications]
type = "smtp"
device_name = "Kitchen"
timezone = "America/Los_Angeles"
host = "smtp.example.com"
port = 587
sender = "monitor@example.com"
recipient = "family@example.com"
username = "monitor@example.com"
password_env = "TEST_SMTP_PASSWORD"
starttls = true
"""


def test_app_lab_zip_contains_pipeline(tmp_path: Path) -> None:
    zip_path = _MODULE.package_app(tmp_path)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        board_config = tomllib.loads(archive.read("python/sound-uno-q.toml").decode())

    assert zip_path.name == "private-sound-alerts.zip"
    assert "app.yaml" in names
    assert "python/main.py" in names
    assert "python/edge_ai/__init__.py" in names
    assert "sketch/sketch.ino" in names
    assert "sketch/sketch.yaml" in names
    assert "python/smtp-password" not in names
    assert board_config["notifications"]["type"] == "none"


def test_email_zip_bundles_password_file_outside_config(tmp_path: Path) -> None:
    private = tmp_path / "private.toml"
    private.write_text(_PRIVATE_CONFIG, encoding="utf-8")

    zip_path = _MODULE.package_app(
        tmp_path / "dist", notifications_config=private, password="test-only-secret"
    )

    with zipfile.ZipFile(zip_path) as archive:
        config_text = archive.read("python/sound-uno-q.toml").decode()
        secret = archive.read("python/smtp-password").decode()
    assert zip_path.name == "private-sound-alerts-email.zip"
    assert "test-only-secret" not in config_text
    assert secret.strip() == "test-only-secret"
    notifications = tomllib.loads(config_text)["notifications"]
    assert notifications["type"] == "smtp"
    assert notifications["password_file"] == "smtp-password"
    assert "password_env" not in notifications
    assert tomllib.loads(config_text)["hardware"]["type"] == "uno_q"

    board_python = tmp_path / "dist" / "private-sound-alerts-email" / "python"
    configured = load_notification_config(board_python / "sound-uno-q.toml")
    assert configured.notifier is not None
    assert configured.notifier.password == "test-only-secret"
    assert configured.notifier.recipient == "family@example.com"


_LIVE_CONFIG = """
[runtime]
[input]
type = "arduino_microphone"
[preprocessing]
type = "audio_waveform"
[inference]
type = "yamnet"
model = "../models/yamnet.onnx"
[decision]
type = "default"
[hardware]
type = "uno_q"
[notifications]
type = "none"
"""


def _live_config(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "yamnet.onnx").write_bytes(b"model-bytes")
    (models / "yamnet_class_map.csv").write_text("index,mid,display_name\n", encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "live.toml"
    config.write_text(_LIVE_CONFIG, encoding="utf-8")
    return config


def test_live_zip_bundles_model_and_onnxruntime(tmp_path: Path) -> None:
    zip_path = _MODULE.package_app(tmp_path / "dist", config_path=_live_config(tmp_path))

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        board_config = tomllib.loads(archive.read("python/sound-uno-q.toml").decode())
        requirements = archive.read("python/requirements.txt").decode()
        manifest = archive.read("app.yaml").decode()

    assert zip_path.name == "private-sound-alerts-live.zip"
    assert "python/models/yamnet.onnx" in names
    assert "python/models/yamnet_class_map.csv" in names
    assert board_config["inference"]["model"] == "models/yamnet.onnx"
    assert "onnxruntime" in requirements
    assert "(live mic)" in manifest


def test_simulated_input_model_zip_is_not_labelled_live(tmp_path: Path) -> None:
    config = _live_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'type = "arduino_microphone"', 'type = "simulated_sound"'
        ),
        encoding="utf-8",
    )

    zip_path = _MODULE.package_app(tmp_path / "dist", config_path=config)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        manifest = archive.read("app.yaml").decode()

    assert zip_path.name == "private-sound-alerts-yamnet-sim.zip"
    assert "python/models/yamnet.onnx" in names
    assert "(YAMNet, simulated mic)" in manifest


def test_live_zip_reports_missing_model(tmp_path: Path) -> None:
    config = _live_config(tmp_path)
    (tmp_path / "models" / "yamnet.onnx").unlink()

    with pytest.raises(ValueError, match="model not found"):
        _MODULE.package_app(tmp_path / "dist", config_path=config)


def test_scheduled_audio_zip_bundles_both_nested_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "yamnet.onnx").write_bytes(b"environment")
    (models / "help-kws.onnx").write_bytes(b"keyword")
    (models / "yamnet_class_map.csv").write_text(
        "index,mid,display_name\n", encoding="utf-8"
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "help.toml"
    config.write_text(
        _LIVE_CONFIG.replace(
            '[inference]\ntype = "yamnet"\nmodel = "../models/yamnet.onnx"',
            '''[inference]
type = "scheduled_audio"
environment_every_steps = 3
[inference.environment]
type = "yamnet"
model = "../models/yamnet.onnx"
[inference.keyword]
type = "keyword_spotter"
model = "../models/help-kws.onnx"''',
        ),
        encoding="utf-8",
    )

    zip_path = _MODULE.package_app(tmp_path / "dist", config_path=config)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        packaged = tomllib.loads(archive.read("python/sound-uno-q.toml").decode())
    assert "python/models/yamnet.onnx" in names
    assert "python/models/help-kws.onnx" in names
    assert "python/models/yamnet_class_map.csv" in names
    assert packaged["inference"]["environment"]["model"] == "models/yamnet.onnx"
    assert packaged["inference"]["keyword"]["model"] == "models/help-kws.onnx"


def test_email_zip_requires_password(tmp_path: Path) -> None:
    private = tmp_path / "private.toml"
    private.write_text(_PRIVATE_CONFIG, encoding="utf-8")

    with pytest.raises(ValueError, match="SMTP password is required"):
        _MODULE.package_app(tmp_path / "dist", notifications_config=private, password=None)


def test_yamnet_zip_bundles_model_labels_and_synthetic_wavs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "yamnet.onnx"
    model.write_bytes(b"test-model")
    monkeypatch.setattr(_MODULE, "YAMNET_MODEL_SOURCE", model)

    zip_path = _MODULE.package_app(tmp_path / "dist", mode="yamnet-test")

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        config = tomllib.loads(archive.read("python/sound-uno-q.toml").decode())
        requirements = archive.read("python/requirements.txt").decode()

    assert zip_path.name == "private-sound-alerts-yamnet-test.zip"
    archive_names = {
        "python/models/yamnet.onnx",
        "python/models/yamnet_class_map.csv",
        "python/samples/background.wav",
        "python/samples/smoke_alarm.wav",
        "python/samples/glass_break.wav",
        "python/samples/fall_thud.wav",
    }
    assert archive_names <= names
    assert config["input"]["type"] == "wav"
    assert config["inference"]["type"] == "yamnet"
    assert config["inference"]["model"] == "models/yamnet.onnx"
    assert "onnxruntime==1.30.0" in requirements
