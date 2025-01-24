from dataclasses import dataclass
from typing import Self

from rich.segment import Segment
from rich.style import Style
from textual import on
from textual.app import App, ComposeResult
from textual.geometry import Region, Size
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget


class Canvas(Widget):
    @dataclass
    class Resize(Message):
        canvas: "Canvas"
        size: Size

    _canvas_size: Size | None = None

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        name=None,
        id=None,
        classes=None,
        disabled=False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        if width is not None and height is not None:
            self.reset(size=Size(width, height), refresh=False)

    def _on_resize(self, event: Resize) -> None:
        self.post_message(self.Resize(canvas=self, size=event.size))

    def reset(self, size: Size | None = None, refresh: bool = True) -> None:
        if size:
            self._canvas_size = size
        if refresh:
            self.refresh()

    def render_lines(self, crop: Region) -> list[Strip]:
        if self._canvas_size is None:
            return []
        return super().render_lines(crop)

    def render_line(self, y: int) -> Strip:
        assert self._canvas_size is not None
        if y < self._canvas_size.height:
            return Strip(
                [
                    Segment(
                        self._canvas_size.width * "/",
                        style=Style.parse("blue on black"),
                    )
                ]
            )
        else:
            return Strip([])


class DemoApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Canvas(10, 5)

    # @on(Canvas.Resize)
    # def redraw(self, event: Canvas.Resize) -> None:
    #     event.canvas.reset(size=event.size)


if __name__ == "__main__":
    DemoApp().run()
