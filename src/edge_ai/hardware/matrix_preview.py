"""Desktop preview of the UNO Q's 13x8 LED matrix, including grayscale alert blinking."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.icons import (
    BLINK_PHASE_SECONDS,
    ICONS,
    LEVEL_BRIGHT,
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    Frame,
    alert_frame,
)

_OFF_RGB = (0x25, 0x30, 0x25)
_ON_RGB = (0x8D, 0xFF, 0x65)


def _colour(level: int) -> str:
    fraction = level / LEVEL_BRIGHT
    red, green, blue = (
        round(off + (on - off) * fraction) for off, on in zip(_OFF_RGB, _ON_RGB)
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


class MatrixPreviewHardware(HardwareBackend):
    """Render the board display locally with Tk, without requiring a UNO Q."""

    WIDTH = MATRIX_WIDTH
    HEIGHT = MATRIX_HEIGHT

    def __init__(
        self,
        *,
        tk_module: Any | None = None,
        pixel_size: int = 28,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(pixel_size, bool) or pixel_size < 8:
            raise ValueError("matrix preview pixel_size must be at least 8")
        if tk_module is None:
            try:
                import tkinter as tk_module
            except ImportError as exc:
                raise RuntimeError("matrix preview requires Python's tkinter module") from exc
        self._tk = tk_module
        self._clock = clock
        try:
            self._root = tk_module.Tk()
        except Exception as exc:
            raise RuntimeError("could not open the matrix preview window") from exc
        self._root.title("UNO Q LED matrix preview")
        self._root.resizable(False, False)
        canvas_size = pixel_size * self.WIDTH
        self._canvas = tk_module.Canvas(
            self._root,
            width=canvas_size,
            height=pixel_size * self.HEIGHT,
            background="#101510",
            highlightthickness=0,
        )
        self._canvas.pack()
        self._pixels: list[list[int]] = []
        for y in range(self.HEIGHT):
            row: list[int] = []
            for x in range(self.WIDTH):
                margin = max(2, pixel_size // 8)
                row.append(
                    self._canvas.create_oval(
                        x * pixel_size + margin,
                        y * pixel_size + margin,
                        (x + 1) * pixel_size - margin,
                        (y + 1) * pixel_size - margin,
                        fill=_colour(0),
                        outline="",
                    )
                )
            self._pixels.append(row)
        self._alert_active = False
        self._alert_event: str | None = None
        self._alert_started = 0.0
        self._delivered = False
        self._drawn_icon_bright: bool | None = None
        self._closed = False
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._draw(self._blank())

    def _blank(self) -> Frame:
        return tuple((0,) * self.WIDTH for _ in range(self.HEIGHT))

    def _on_close(self) -> None:
        self._closed = True
        self._root.destroy()

    def _refresh(self) -> None:
        if self._closed:
            raise RuntimeError("matrix preview window was closed")
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception as exc:
            self._closed = True
            raise RuntimeError("matrix preview window was closed") from exc

    def _draw(self, frame: Frame) -> None:
        for y, row in enumerate(frame):
            for x, level in enumerate(row):
                self._canvas.itemconfigure(self._pixels[y][x], fill=_colour(level))
        self._refresh()

    def _draw_alert(self, *, force: bool = False) -> None:
        if self._alert_event is None:
            return
        elapsed = self._clock() - self._alert_started
        icon_bright = int(elapsed / BLINK_PHASE_SECONDS) % 2 == 0
        if force or icon_bright != self._drawn_icon_bright:
            self._drawn_icon_bright = icon_bright
            self._draw(
                alert_frame(self._alert_event, icon_bright=icon_bright, delivered=self._delivered)
            )

    def check_connection(self) -> None:
        self._refresh()

    def set_led(self, enabled: bool) -> None:
        # This preview represents the matrix only; its title exposes the built-in LED state.
        self._root.title(f"UNO Q LED matrix preview — {'alert' if enabled else 'idle'}")
        self._refresh()

    def set_pwm(self, channel: int, value: float) -> None:
        raise RuntimeError("matrix preview does not provide PWM")

    def move_servo(self, channel: int, degrees: float) -> None:
        raise RuntimeError("matrix preview does not provide servo control")

    def show_alert(self, event: str) -> None:
        if event not in ICONS:
            raise ValueError(f"unsupported visual alert: {event!r}")
        changed = event != self._alert_event
        if changed:
            self._alert_event = event
            self._alert_started = self._clock()
            self._delivered = False
        self._alert_active = True
        self._draw_alert(force=changed)
        self.set_led(True)

    def show_notification_status(self, delivered: bool) -> None:
        if self._alert_event is None:
            return
        self._delivered = delivered
        self._draw_alert(force=True)

    def clear_alert(self) -> None:
        self._alert_active = False
        self._alert_event = None
        self._delivered = False
        self._drawn_icon_bright = None
        self._draw(self._blank())
        self.set_led(False)

    def show_spectrum(self, columns: tuple[int, ...]) -> None:
        if len(columns) != self.WIDTH or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= self.HEIGHT
            for value in columns
        ):
            raise ValueError("audio spectrum must contain 13 integer levels from 0 through 8")
        if self._alert_active:
            self._draw_alert()
            return
        self._draw(
            tuple(
                tuple(
                    LEVEL_BRIGHT if self.HEIGHT - 1 - y < height else 0 for height in columns
                )
                for y in range(self.HEIGHT)
            )
        )

    def apply_decision(self, decision: Decision) -> None:
        if decision.action == "alert":
            if decision.event is None:
                self.set_led(True)
                self._alert_active = True
            else:
                self.show_alert(decision.event)
        else:
            self.clear_alert()

    def shutdown(self) -> None:
        if not self._closed:
            self._on_close()
