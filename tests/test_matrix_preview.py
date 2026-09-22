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
    preview = MatrixPreviewHardware(tk_module=FakeTk, pixel_size=8)

    preview.show_spectrum((0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4))
    assert len(preview._canvas.fills) == 104  # type: ignore[attr-defined]
    preview.apply_decision(Decision("alert", event="smoke_alarm"))
    before = dict(preview._canvas.fills)  # type: ignore[attr-defined]
    preview.show_spectrum((8,) * 13)

    assert preview._canvas.fills == before  # type: ignore[attr-defined]
    preview.shutdown()
