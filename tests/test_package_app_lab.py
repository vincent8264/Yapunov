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


def test_email_zip_requires_password(tmp_path: Path) -> None:
    private = tmp_path / "private.toml"
    private.write_text(_PRIVATE_CONFIG, encoding="utf-8")

    with pytest.raises(ValueError, match="SMTP password is required"):
        _MODULE.package_app(tmp_path / "dist", notifications_config=private, password=None)
