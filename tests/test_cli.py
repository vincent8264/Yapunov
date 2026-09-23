from pathlib import Path

import pytest

from edge_ai import cli
from edge_ai.config import ConfiguredNotifier
from edge_ai.notifications import AlertNotification, Notifier


class RecordingNotifier(Notifier):
    channel = "email"

    def __init__(self) -> None:
        self.alerts: list[AlertNotification] = []
        self.closed = False

    def notify(self, alert: AlertNotification) -> None:
        self.alerts.append(alert)

    def close(self) -> None:
        self.closed = True


def test_notification_command_sends_one_preset_alert(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notifier = RecordingNotifier()
    monkeypatch.setattr(
        cli,
        "load_notification_config",
        lambda path: ConfiguredNotifier(
            notifier=notifier,
            device_name="Living room",
            local_timezone=cli.datetime.now().astimezone().tzinfo,
        ),
    )

    result = cli.main(
        [
            "test-notification",
            "--config",
            "private.toml",
            "--event",
            "fall_thud",
        ]
    )

    assert result == 0
    assert len(notifier.alerts) == 1
    assert notifier.alerts[0].event == "fall_thud"
    assert notifier.alerts[0].device_name == "Living room"
    assert notifier.closed
    assert "Test notification delivered" in capsys.readouterr().out


def test_notification_command_rejects_disabled_delivery(tmp_path: Path) -> None:
    path = tmp_path / "disabled.toml"
    path.write_text('[notifications]\ntype = "none"\n', encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli.main(["test-notification", "--config", str(path)])

    assert error.value.code == 2


def test_setup_command_starts_local_portal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str, int]] = []
    monkeypatch.setattr(
        cli,
        "run_setup_server",
        lambda path, host, port: calls.append((path, host, port)),
    )

    result = cli.main(
        [
            "setup",
            "--config",
            "private.toml",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
        ]
    )

    assert result == 0
    assert calls == [(Path("private.toml"), "0.0.0.0", 8123)]


def test_inspect_audio_command_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path, Path, int]] = []
    monkeypatch.setattr(cli, "_inspect_audio", lambda *args: calls.append(args))

    result = cli.main(
        [
            "inspect-audio",
            "sample.wav",
            "--model",
            "yamnet.onnx",
            "--class-map",
            "labels.csv",
            "--top-k",
            "4",
        ]
    )

    assert result == 0
    assert calls == [(Path("sample.wav"), Path("yamnet.onnx"), Path("labels.csv"), 4)]


def test_evaluate_keyword_command_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path, float, int]] = []
    monkeypatch.setattr(cli, "_evaluate_keyword", lambda *args: calls.append(args))

    result = cli.main(
        [
            "evaluate-keyword",
            "--dataset",
            "clips",
            "--model",
            "help.onnx",
            "--threshold",
            "0.7",
            "--positive-index",
            "0",
        ]
    )

    assert result == 0
    assert calls == [(Path("clips"), Path("help.onnx"), 0.7, 0)]


def test_caption_audio_command_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "_caption_audio", lambda *args: calls.append(args))

    result = cli.main(
        [
            "caption-audio",
            "one.wav",
            "two.wav",
            "--model-dir",
            "models/caption",
            "--style",
            "audioset",
            "--max-new-tokens",
            "24",
        ]
    )

    assert result == 0
    assert calls == [
        ([Path("one.wav"), Path("two.wav")], Path("models/caption"), "audioset", 24)
    ]
