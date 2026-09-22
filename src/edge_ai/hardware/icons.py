"""Portable 8x8 icon artwork for the sound-alert display."""

from typing import Final

Icon = tuple[str, str, str, str, str, str, str, str]

ICONS: Final[dict[str, Icon]] = {
    "smoke_alarm": (
        "...##...",
        "..####..",
        ".######.",
        ".######.",
        ".######.",
        "########",
        "...##...",
        "...##...",
    ),
    "glass_break": (
        "#......#",
        ".#..#.#.",
        "..##.#..",
        "...##...",
        "..#.##..",
        ".#.#..#.",
        "#......#",
        "########",
    ),
    "fall_thud": (
        "...##...",
        "...##...",
        "..####..",
        "...##...",
        "..###...",
        ".#..##..",
        "#....##.",
        "########",
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
}


def icon_rows(event: str) -> tuple[int, ...]:
    """Return eight row bitmasks suitable for a matrix-specific adapter."""
    try:
        icon = ICONS[event]
    except KeyError as exc:
        raise ValueError(f"no 8x8 icon is defined for event {event!r}") from exc
    return tuple(int(row.replace("#", "1").replace(".", "0"), 2) for row in icon)


def render_icon(event: str) -> str:
    """Render an icon as a terminal-friendly block preview."""
    try:
        icon = ICONS[event]
    except KeyError as exc:
        raise ValueError(f"no 8x8 icon is defined for event {event!r}") from exc
    return "\n".join(row.replace("#", "██").replace(".", "  ") for row in icon)
