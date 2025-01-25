import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import floor

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import on
from textual._box_drawing import BOX_CHARACTERS
from textual.app import App, ComposeResult
from textual.geometry import Region, Size
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget


class TextAlign(enum.Enum):
    LEFT = enum.auto()
    CENTER = enum.auto()
    RIGHT = enum.auto()


class Canvas(Widget):
    @dataclass
    class Resize(Message):
        canvas: "Canvas"
        size: Size

    _canvas_size: Size | None = None
    get_box = BOX_CHARACTERS.__getitem__

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

        if self._canvas_size:
            self._buffer = [
                ["." for _ in range(self._canvas_size.width)]
                for _ in range(self._canvas_size.height)
            ]
            self._styles = [
                ["" for _ in range(self._canvas_size.width)]
                for _ in range(self._canvas_size.height)
            ]

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

    def set_pixel(self, x: int, y: int, char="█", style="white") -> None:
        if (
            x < 0
            or y < 0
            or x >= self._canvas_size.width
            or y >= self._canvas_size.height
        ):
            # coordinates are outside canvas
            return

        self._buffer[y][x] = char
        self._styles[y][x] = style
        self.refresh()

    def set_pixels(
        self, coordinates: Iterable[tuple[int, int]], char="█", style="white"
    ) -> None:
        for x, y in coordinates:
            self.set_pixel(x, y, char, style)

    def draw_line(
        self, x0: int, y0: int, x1: int, y1: int, char="█", style="white"
    ) -> None:
        self.set_pixels(self._get_line_coordinates(x0, y0, x1, y1), char, style)

    def draw_rectangle_box(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        thickness: int = 1,
        style: str = "white",
    ) -> None:
        T = thickness
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        self.set_pixel(x0, y0, char=self.get_box((0, T, T, 0)), style=style)
        self.set_pixel(x1, y0, char=self.get_box((0, 0, T, T)), style=style)
        self.set_pixel(x1, y1, char=self.get_box((T, 0, 0, T)), style=style)
        self.set_pixel(x0, y1, char=self.get_box((T, T, 0, 0)), style=style)
        for y in y0, y1:
            self.draw_line(
                x0 + 1, y, x1 - 1, y, char=self.get_box((0, T, 0, T)), style=style
            )
        for x in x0, x1:
            self.draw_line(
                x, y0 + 1, x, y1 - 1, char=self.get_box((T, 0, T, 0)), style=style
            )

    def write_text(
        self,
        x: int,
        y: int,
        text: str,
        align: TextAlign = TextAlign.LEFT,
        style: str = "white",
    ) -> str:
        if y < 0 or y >= self._canvas_size.height:
            return

        # parse markup
        rich_text = Text.from_markup(text)
        # store plain text
        plain_text = rich_text.plain
        # store styles for each individual character
        rich_styles = []
        for c in rich_text.divide(range(1, len(plain_text))):
            style = Style()
            for span in c._spans:
                style += Style.parse(span.style)
            rich_styles.append(style)

        if align == TextAlign.RIGHT:
            x -= len(plain_text) - 1
        elif align == TextAlign.CENTER:
            div, mod = divmod(len(plain_text), 2)
            x -= div
            if mod == 0:
                # even number of characters, shift one to the right since I just
                # like that better -- DF
                x += 1

        if x <= -len(plain_text) or x >= self._canvas_size.width:
            # no part of text falls inside the canvas
            return

        overflow_left = -x
        overflow_right = x + len(plain_text) - self._canvas_size.width
        if overflow_left > 0:
            buffer_left = 0
            text_left = overflow_left
        else:
            buffer_left = x
            text_left = 0
        if overflow_right > 0:
            buffer_right = None
            text_right = -overflow_right
        else:
            buffer_right = x + len(plain_text)
            text_right = None

        self._buffer[y][buffer_left:buffer_right] = plain_text[text_left:text_right]
        self._styles[y][buffer_left:buffer_right] = [
            str(s) for s in rich_styles[text_left:text_right]
        ]
        assert len(self._buffer[y]) == self._canvas_size.width
        assert len(self._styles[y]) == self._canvas_size.width
        self.refresh()

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
    _bx = 0
    _bdx = 1
    _by = 0
    _bdy = 1
    _tidx = 0

    def compose(self) -> ComposeResult:
        yield Canvas(40, 20)

    def on_mount(self) -> None:
        self.set_interval(1 / 10, self.redraw_canvas)

    @on(Canvas.Resize)
    def resize(self, event: Canvas.Resize) -> None:
        event.canvas.reset(size=event.size)

    def redraw_canvas(self) -> None:
        canvas = self.query_one(Canvas)
        canvas.reset()
        canvas.draw_line(0, 0, 8, 8)
        canvas.draw_line(0, 19, 39, 0, char="X", style="red")
        canvas.write_text(
            self._tidx,
            10,
            "[green]This text is [bold]easy[/bold] to read",
        )
        canvas.draw_rectangle_box(
            self._bx, self._by, self._bx + 20, self._by + 10, thickness=2
        )
        self._bx += self._bdx
        if (self._bx <= 0) or (self._bx + 20 >= canvas.size.width - 1):
            self._bdx *= -1
        self._by += self._bdy
        if (self._by <= 0) or (self._by + 10 >= canvas.size.height - 1):
            self._bdy *= -1
        self._tidx += 2
        if self._tidx >= canvas.size.width + 20:
            self._tidx = -20


class MapDemoApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Canvas(20, 20)

    def on_mount(self) -> None:
        canvas = self.query_one(Canvas)
        canvas.draw_rectangle_box(0, 0, 5, 21)
        x = [-0.01, 0.01, 2.49, 2.51, 4.99, 5.01, 7.49, 7.51, 9.9, 10.0, 10.01]
        y = [i for i in range(len(x))]
        mapped_x = [map_coordinate_to_pixel(u, 0.0, 10.0, 1, 4) for u in x]
        canvas.set_pixels(((a, b) for a, b in zip(mapped_x, y)))
        for idx, value in enumerate(x):
            canvas.write_text(x=10, y=idx, text=str(value))


def map_coordinate_to_pixel(
    x: float, a: float, b: float, first_pixel: int, last_pixel: int
) -> int:
    return floor(linear_mapper(x, a, b, first_pixel, last_pixel + 1))


def linear_mapper(
    x: float | int,
    a: float | int,
    b: float | int,
    a_prime: float | int,
    b_prime: float | int,
) -> float:
    return a_prime + (x - a) * (b_prime - a_prime) / (b - a)


if __name__ == "__main__":
    # DemoApp().run()
    MapDemoApp().run()
