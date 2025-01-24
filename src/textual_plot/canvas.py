import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

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
        try:
            self._buffer[y][x] = char
            self._styles[y][x] = style
        except IndexError:
            pass

    def set_pixels(
        self, coordinates: Iterable[tuple[int, int]], char: str, style: str
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
        yield (canvas := Canvas(60, 20))
        canvas.draw_line(0, 0, 8, 8)
        canvas.draw_line(0, 19, 39, 0, char="X", style="red")
        canvas.draw_rectangle_box(2, 12, 40, 4, thickness=2)
        canvas.write_text(-2, 0, "".join([str(i % 10) for i in range(70)]))
        canvas.write_text(
            -2,
            1,
            "[italic bold purple]Hi[/] [bold]there[/bold], are you [green]o[bold]k[/green]?[/]?",
        )
        canvas.write_text(
            1,
            2,
            "[italic bold purple]Hi[/] [bold]there[/bold], are you [green]o[bold]k[/green]?[/]?",
        )
        canvas.write_text(
            30,
            1,
            "[italic bold purple]Hi[/] [bold]there[/bold], are you [green]o[bold]k[/green]?[/]?",
        )
        canvas.write_text(
            38,
            1,
            "[italic bold purple]Hi[/] [bold]there[/bold], are you [green]o[bold]k[/green]?[/]?",
        )

    # @on(Canvas.Resize)
    # def redraw(self, event: Canvas.Resize) -> None:
    # canvas = event.canvas
    # canvas.reset(size=event.size)


if __name__ == "__main__":
    DemoApp().run()
