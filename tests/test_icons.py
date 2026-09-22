import pytest

from edge_ai.hardware.icons import ICONS, icon_rows, render_icon


def test_every_sound_event_has_a_distinct_8x8_icon() -> None:
    events = ("smoke_alarm", "glass_break", "fall_thud")
    rows = [icon_rows(event) for event in events]

    assert len(set(rows)) == 3
    assert all(len(item) == 8 for item in rows)
    assert all(0 <= row <= 255 for item in rows for row in item)
    assert all(len(line) == 8 for event in events for line in ICONS[event])
    assert "██" in render_icon("smoke_alarm")


def test_unknown_icon_has_specific_error() -> None:
    with pytest.raises(ValueError, match="unknown"):
        icon_rows("unknown")
