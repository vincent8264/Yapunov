"""Portable 8x8 icon artwork for the sound-alert display."""

from typing import Final

Icon = tuple[str, str, str, str, str, str, str, str]

ICONS: Final[dict[str, Icon]] = {
    # Three wavy smoke wisps rising from the bottom to the top.
    "smoke_alarm": (
        ".#..#..#",
        ".#..#..#",
        "#..#..#.",
        "#..#..#.",
        ".#..#..#",
        ".#..#..#",
        "#..#..#.",
        "#..#..#.",
    ),
    # A lightning bolt.
    "glass_break": (
        "...####.",
        "..####..",
        ".####...",
        "#######.",
        "...###..",
        "..###...",
        ".##.....",
        "#.......",
    ),
    # An upside-down person: legs at the top, head at the bottom.
    "fall_thud": (
        "#......#",
        ".#....#.",
        "..#..#..",
        "...##...",
        "#..##..#",
        ".######.",
        "...##...",
        "...##...",
    ),
    "help_call": (
        "##....##",
        "##....##",
        "##....##",
        "########",
        "########",
        "##....##",
        "##....##",
        "##....##",
    ),
    # A microphone capsule and stand crossed by a descending fault slash.
    "microphone_fault": (
        "...##..#",
        "..####.#",
        "..#####.",
        "..####..",
        "#.####.#",
        ".######.",
        ".#.##...",
        "#.####..",
    ),
}


def icon_rows(event: str) -> tuple[int, ...]:
    """Return eight row bitmasks suitable for a matrix-specific adapter."""
    try:
        icon = ICONS[event]
    except KeyError as exc:
        raise ValueError(f"no 8x8 icon is defined for event {event!r}") from exc
    return tuple(int(row.replace("#", "1").replace(".", "0"), 2) for row in icon)


MATRIX_WIDTH: Final = 13
MATRIX_HEIGHT: Final = 8
ICON_AREA_WIDTH: Final = 12
ICON_OFFSET: Final = 2
STATUS_COLUMN: Final = 12
LEVEL_OFF: Final = 0
LEVEL_DIM: Final = 1
LEVEL_BRIGHT: Final = 7
BLINK_PHASE_SECONDS: Final = 0.5

Frame = tuple[tuple[int, ...], ...]


def alert_frame(event: str, *, icon_bright: bool, delivered: bool) -> Frame:
    """Return 8 rows of 13 brightness levels for one blink phase of an alert.

    Columns 0-11 hold the icon, with a background that blinks opposite to it. Column
    12 is a solid bar only when the alert email was delivered.
    """
    try:
        icon = ICONS[event]
    except KeyError as exc:
        raise ValueError(f"no 8x8 icon is defined for event {event!r}") from exc
    icon_level = LEVEL_BRIGHT if icon_bright else LEVEL_DIM
    background_level = LEVEL_DIM if icon_bright else LEVEL_BRIGHT
    status_level = LEVEL_BRIGHT if delivered else LEVEL_OFF
    rows = []
    for icon_row in icon:
        row = [background_level] * ICON_AREA_WIDTH + [status_level]
        for x, pixel in enumerate(icon_row):
            if pixel == "#":
                row[x + ICON_OFFSET] = icon_level
        rows.append(tuple(row))
    return tuple(rows)


def render_icon(event: str) -> str:
    """Render an icon as a terminal-friendly block preview."""
    try:
        icon = ICONS[event]
    except KeyError as exc:
        raise ValueError(f"no 8x8 icon is defined for event {event!r}") from exc
    return "\n".join(row.replace("#", "██").replace(".", "  ") for row in icon)
