from pathlib import Path
import re

import pytest

from edge_ai.hardware.icons import (
    ICONS,
    LEVEL_BRIGHT,
    LEVEL_DIM,
    LEVEL_OFF,
    alert_frame,
    icon_rows,
    render_icon,
)


def test_every_event_and_fault_has_a_distinct_8x8_icon() -> None:
    events = (
        "smoke_alarm",
        "glass_break",
        "fall_thud",
        "help_call",
        "microphone_fault",
    )
    rows = [icon_rows(event) for event in events]

    assert len(set(rows)) == len(events)
    assert all(len(item) == 8 for item in rows)
    assert all(0 <= row <= 255 for item in rows for row in item)
    assert all(len(line) == 8 for event in events for line in ICONS[event])
    assert "██" in render_icon("smoke_alarm")


def test_sketch_icon_bitmaps_match_python_artwork() -> None:
    sketch = (
        Path(__file__).resolve().parents[1] / "app_lab" / "starter_app" / "sketch" / "sketch.ino"
    ).read_text()
    for event, name in (
        ("smoke_alarm", "ICON_SMOKE"),
        ("glass_break", "ICON_GLASS"),
        ("fall_thud", "ICON_FALL"),
        ("help_call", "ICON_HELP"),
        ("microphone_fault", "ICON_MICROPHONE_FAULT"),
    ):
        match = re.search(rf"{name}\[8\] = \{{([^}}]*)\}}", sketch)
        assert match is not None, name
        rows = tuple(int(value, 16) for value in match.group(1).split(","))
        assert rows == icon_rows(event), event


def test_unknown_icon_has_specific_error() -> None:
    with pytest.raises(ValueError, match="unknown"):
        icon_rows("unknown")
    with pytest.raises(ValueError, match="unknown"):
        alert_frame("unknown", icon_bright=True, delivered=False)


def test_alert_frame_blinks_icon_opposite_to_background_in_12_columns() -> None:
    bright = alert_frame("smoke_alarm", icon_bright=True, delivered=False)
    dim = alert_frame("smoke_alarm", icon_bright=False, delivered=False)

    assert len(bright) == 8 and all(len(row) == 13 for row in bright)
    for y, icon_row in enumerate(ICONS["smoke_alarm"]):
        for x in range(12):
            icon_x = x - 2
            is_icon = 0 <= icon_x < 8 and icon_row[icon_x] == "#"
            assert bright[y][x] == (LEVEL_BRIGHT if is_icon else LEVEL_DIM)
            assert dim[y][x] == (LEVEL_DIM if is_icon else LEVEL_BRIGHT)
    assert all(row[12] == LEVEL_OFF for row in bright + dim)


def test_alert_frame_status_column_is_a_bar_only_when_delivered() -> None:
    delivered = alert_frame("glass_break", icon_bright=False, delivered=True)
    pending = alert_frame("glass_break", icon_bright=False, delivered=False)

    assert [row[12] for row in delivered] == [LEVEL_BRIGHT] * 8
    assert [row[12] for row in pending] == [LEVEL_OFF] * 8
    assert [row[:12] for row in delivered] == [row[:12] for row in pending]
