from collections.abc import Iterable, Iterator
from dataclasses import dataclass

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
            self._buffer = [
                ["." for _ in range(size.width)] for _ in range(size.height)
            ]
            self._styles = [["" for _ in range(size.width)] for _ in range(size.height)]

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
                    Segment(char, style=Style.parse(style))
                    for char, style in zip(self._buffer[y], self._styles[y])
                ]
            )
        else:
            return Strip([])

    def set_pixel(self, x: int, y: int, char: str, style: str) -> None:
        self._buffer[y][x] = char
        self._styles[y][x] = style

    def set_pixels(
        self, coordinates: Iterable[tuple[int, int]], char: str, style: str
    ) -> None:
        for x, y in coordinates:
            self.set_pixel(x, y, char, style)

    def draw_line(
        self, x0: int, y0: int, x1: int, y1: int, char="█", style="white"
    ) -> None:
        self.set_pixels(self._get_line_coordinates(x0, y0, x1, y1), char, style)

    def _get_line_coordinates(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> Iterator[tuple[int, int]]:
        """Get all pixel coordinates on the line between two points.

        Algorithm was taken from
        https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm and
        translated to Python.

        Args:
            x0: starting point x coordinate
            y0: starting point y coordinate
            x1: end point x coordinate
            y1: end point y coordinate

        Yields:
            Tuples of (x, y) coordinates that make up the line.
        """
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        while True:
            yield x0, y0
            e2 = 2 * error
            if e2 >= dy:
                if x0 == x1:
                    break
                error = error + dy
                x0 = x0 + sx
            if e2 <= dx:
                if y0 == y1:
                    break
                error = error + dx
                y0 = y0 + sy


class DemoApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Canvas(40, 20)

    def on_mount(self) -> None:
        canvas = self.query_one(Canvas)
        canvas.draw_line(0, 0, 8, 8)
        canvas.draw_line(0, 19, 39, 0, char="X", style="red")

    # @on(Canvas.Resize)
    # def redraw(self, event: Canvas.Resize) -> None:
    #     event.canvas.reset(size=event.size)


if __name__ == "__main__":
    DemoApp().run()
