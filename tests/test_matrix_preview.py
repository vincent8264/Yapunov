from edge_ai.decision import Decision
from edge_ai.hardware.matrix_preview import MatrixPreviewHardware


class FakeRoot:
    def title(self, value: str) -> None:
        self.last_title = value

    def resizable(self, *_: object) -> None:
        pass

    def protocol(self, *_: object) -> None:
        pass

    def update_idletasks(self) -> None:
        pass

    def update(self) -> None:
        pass

    def destroy(self) -> None:
        self.destroyed = True


class FakeCanvas:
    def __init__(self, *_: object, **__: object) -> None:
        self.fills: dict[int, str] = {}
        self._next = 0

    def pack(self) -> None:
        pass

    def create_oval(self, *_: object, **__: object) -> int:
        self._next += 1
        return self._next

    def itemconfigure(self, item: int, **kwargs: str) -> None:
        self.fills[item] = kwargs["fill"]


class FakeTk:
    Canvas = FakeCanvas

    @staticmethod
    def Tk() -> FakeRoot:
        return FakeRoot()


def test_matrix_preview_draws_spectrum_and_preserves_alert_priority() -> None:
    preview = MatrixPreviewHardware(tk_module=FakeTk, pixel_size=8, clock=lambda: 0.0)

    preview.show_spectrum((0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4))
    assert len(preview._canvas.fills) == 104  # type: ignore[attr-defined]
    preview.apply_decision(Decision("alert", event="smoke_alarm"))
    before = dict(preview._canvas.fills)  # type: ignore[attr-defined]
    preview.show_spectrum((8,) * 13)

    assert preview._canvas.fills == before  # type: ignore[attr-defined]
    preview.shutdown()


def _pixel_fill(preview: MatrixPreviewHardware, x: int, y: int) -> str:
    return preview._canvas.fills[y * 13 + x + 1]  # type: ignore[attr-defined]


def test_matrix_preview_blinks_icon_against_background_and_shows_email_bar() -> None:
    now = [0.0]
    preview = MatrixPreviewHardware(tk_module=FakeTk, pixel_size=8, clock=lambda: now[0])
    preview.show_alert("smoke_alarm")
    # Row 0 of the smoke icon is ".#..#..#", drawn from x=2: x=3 is icon, x=0 is background.
    icon_on, background_dim = _pixel_fill(preview, 3, 0), _pixel_fill(preview, 0, 0)
    status_off = _pixel_fill(preview, 12, 0)

    now[0] = 0.6
    preview.show_spectrum((0,) * 13)

    assert _pixel_fill(preview, 3, 0) == background_dim
    assert _pixel_fill(preview, 0, 0) == icon_on
    assert _pixel_fill(preview, 12, 0) == status_off

    preview.show_notification_status(True)
    assert all(_pixel_fill(preview, 12, y) == icon_on for y in range(8))

    preview.clear_alert()
    assert _pixel_fill(preview, 12, 0) == status_off
    preview.shutdown()


def test_matrix_preview_shows_and_clears_microphone_fault() -> None:
    preview = MatrixPreviewHardware(tk_module=FakeTk, pixel_size=8, clock=lambda: 0.0)

    preview.show_input_fault("unavailable", "USB microphone disappeared")
    assert preview._alert_event == "microphone_fault"  # type: ignore[attr-defined]

    preview.clear_input_fault("unavailable")
    assert preview._alert_event is None  # type: ignore[attr-defined]
    preview.shutdown()
