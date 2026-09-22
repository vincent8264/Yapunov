"""Desktop preview of the UNO Q's 13x8 single-colour LED matrix."""

from __future__ import annotations

from typing import Any

from edge_ai.decision import Decision
from edge_ai.hardware.base import HardwareBackend
from edge_ai.hardware.icons import ICONS


class MatrixPreviewHardware(HardwareBackend):
    """Render the board display locally with Tk, without requiring a UNO Q."""

    WIDTH = 13
    HEIGHT = 8

    def __init__(self, *, tk_module: Any | None = None, pixel_size: int = 28) -> None:
        if isinstance(pixel_size, bool) or pixel_size < 8:
            raise ValueError("matrix preview pixel_size must be at least 8")
        if tk_module is None:
            try:
                import tkinter as tk_module
            except ImportError as exc:
                raise RuntimeError("matrix preview requires Python's tkinter module") from exc
        self._tk = tk_module
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
                        fill="#253025",
                        outline="",
                    )
                )
            self._pixels.append(row)
        self._alert_active = False
        self._closed = False
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._draw([[False] * self.WIDTH for _ in range(self.HEIGHT)])

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

    def _draw(self, frame: list[list[bool]]) -> None:
        for y, row in enumerate(frame):
            for x, lit in enumerate(row):
                self._canvas.itemconfigure(self._pixels[y][x], fill="#8dff65" if lit else "#253025")
        self._refresh()

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
        rows = ICONS[event]
        frame = [[False] * self.WIDTH for _ in range(self.HEIGHT)]
        for y, row in enumerate(rows):
            for x, value in enumerate(row):
                frame[y][x + 2] = value == "#"
        self._alert_active = True
        self._draw(frame)
        self.set_led(True)

    def clear_alert(self) -> None:
        self._alert_active = False
        self._draw([[False] * self.WIDTH for _ in range(self.HEIGHT)])
        self.set_led(False)

    def show_spectrum(self, columns: tuple[int, ...]) -> None:
        if len(columns) != self.WIDTH or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= self.HEIGHT
            for value in columns
        ):
            raise ValueError("audio spectrum must contain 13 integer levels from 0 through 8")
        if self._alert_active:
            return
        frame = [[False] * self.WIDTH for _ in range(self.HEIGHT)]
        for x, height in enumerate(columns):
            for y in range(height):
                frame[self.HEIGHT - 1 - y][x] = True
        self._draw(frame)

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
