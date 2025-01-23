import enum
from dataclasses import dataclass
from typing import Iterator, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.segment import Segment
from rich.style import Style
from textual._box_drawing import BOX_CHARACTERS
from textual.app import App, ComposeResult
from textual.color import Color
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widget import Widget


@dataclass
class DataSet:
    x: NDArray[np.floating]
    y: NDArray[np.floating]


@dataclass
class LinePlot(DataSet):
    line_style: Style


@dataclass
class ScatterPlot(DataSet):
    marker: str
    marker_style: Style


class TextAlign(enum.Enum):
    LEFT = enum.auto()
    CENTER = enum.auto()
    RIGHT = enum.auto()


class PlotWidget(Widget):
    _datasets: list[DataSet]

    _x_min: float = 0.0
    _x_max: float = 10.0
    _y_min: float = 0.0
    _y_max: float = 30.0
    _margin_top: int = 1
    _margin_right: int = 5
    _margin_bottom: int = 5
    _margin_left: int = 10

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.clear()

    def clear(self) -> None:
        self._datasets = []

    def plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        line_style: str = "white",
    ) -> None:
        self._datasets.append(
            LinePlot(x=np.array(x), y=np.array(y), line_style=Style.parse(line_style))
        )
        self.refresh()

    def scatter(
        self, x: ArrayLike, y: ArrayLike, marker: str = "o", marker_style: str = "white"
    ) -> None:
        self._datasets.append(
            ScatterPlot(
                x=np.array(x),
                y=np.array(y),
                marker=marker,
                marker_style=Style.parse(marker_style),
            )
        )
        self.refresh()

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        self._render_plot()
        return super().refresh(
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    def _linear_mapper(
        self,
        x: float | int,
        a: float | int,
        b: float | int,
        a_prime: float | int,
        b_prime: float | int,
    ) -> float:
        # FIXME: needs adjustment because I want 'outer' limits
        return a_prime + (x - a) * (b_prime - a_prime) / (b - a)

    def _map_coordinates_to_pixels(
        self,
        x: list[float] | NDArray[np.floating],
        y: list[float] | NDArray[np.floating],
    ) -> tuple[list[int], list[int]]:
        return (
            [
                round(
                    self._linear_mapper(
                        px, self._x_min, self._x_max, 0, self._plot_size.width - 1
                    )
                )
                for px in x
            ],
            [
                round(
                    self._linear_mapper(
                        py, self._y_min, self._y_max, 0, self._plot_size.height - 1
                    )
                )
                for py in y
            ],
        )

    def _render_plot(self) -> None:
        self._plot_size = self.size - (
            self._margin_left + self._margin_right,
            self._margin_top + self._margin_bottom,
        )

        self._canvas = [
            [Segment(" ") for _ in range(self._plot_size.width)]
            for _ in range(self._plot_size.height)
        ]
        for dataset in self._datasets:
            if isinstance(dataset, ScatterPlot):
                self._render_scatter_plot(dataset)
            elif isinstance(dataset, LinePlot):
                self._render_line_plot(dataset)

    def _render_scatter_plot(self, dataset: ScatterPlot) -> None:
        x, y = self._map_coordinates_to_pixels(dataset.x, dataset.y)

        for px, py in zip(x, y):
            try:
                self._canvas[py][px] = Segment(
                    text=dataset.marker, style=dataset.marker_style
                )
            except IndexError:
                # data point is outside plot area
                pass

    def _render_line_plot(self, dataset: LinePlot) -> None:
        x, y = self._map_coordinates_to_pixels(dataset.x, dataset.y)

        for i in range(1, len(x)):
            for px, py in self.bresenham_line(x[i - 1], y[i - 1], x[i], y[i]):
                try:
                    self._canvas[py][px] = Segment(text="█", style=dataset.line_style)
                except IndexError:
                    # data point is outside plot area
                    pass

    def bresenham_line(
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

    def render_line(self, y: int) -> Strip:
        """Render a line of the widget. y is relative to the top of the widget."""
        # calculate rows and ranges of parts
        rows_plot = (
            self._margin_top,
            self.size.height - self._margin_bottom - 1,
        )
        row_top_border = rows_plot[0] - 1
        row_bottom_border = rows_plot[1] + 1

        # render parts
        if y < row_top_border:
            return self._render_top_margin()
        elif y == row_top_border:
            return self._render_box_top()
        elif rows_plot[0] <= y <= rows_plot[1]:
            return self._render_plot_area_with_leftright_margins(idx=y - rows_plot[0])
        elif y == row_bottom_border:
            return self._render_box_bottom()
        else:
            return self._render_bottom_margin(idx=y - (row_bottom_border + 1))

    def _render_top_margin(self) -> Strip:
        return Strip(
            [Segment(" " * self.size.width, Style(bgcolor="red"))],
            cell_length=self.size.width,
        )

    def _render_box_top(self) -> Strip:
        left_margin = insert_text_into(
            " " * (self._margin_left - 1),
            str(self._y_max),
            index=self._margin_left - 3,
            align=TextAlign.RIGHT,
        )

        get_box = BOX_CHARACTERS.__getitem__
        return Strip(
            [
                Segment(left_margin, Style.parse("bold white on black")),
                Segment(
                    get_box((0, 2, 2, 2))
                    + get_box((0, 2, 0, 2)) * self._plot_size.width
                    + get_box((0, 0, 2, 2))
                ),
                Segment(" " * (self._margin_right - 1), Style(bgcolor="green")),
            ]
        )

    def _render_plot_area_with_leftright_margins(self, idx: int) -> Strip:
        get_box = BOX_CHARACTERS.__getitem__
        left_margin = " " * (self._margin_left - 1)
        return Strip(
            [Segment(left_margin, Style(bgcolor="blue"))]
            + ([Segment(get_box((2, 0, 2, 0)))] if self._margin_left else [])
            + self._canvas[self._plot_size.height - 1 - idx]
            + ([Segment(get_box((2, 0, 2, 0)))] if self._margin_right else [])
            + [
                Segment(" " * (self._margin_right - 1), Style(bgcolor="blue")),
            ],
            cell_length=self.size.width,
        )

    def _render_box_bottom(self) -> Strip:
        left_margin = insert_text_into(
            " " * (self._margin_left - 1),
            str(self._y_min),
            index=self._margin_left - 3,
            align=TextAlign.RIGHT,
        )

        get_box = BOX_CHARACTERS.__getitem__
        return Strip(
            [
                Segment(left_margin, Style.parse("bold white on black")),
                Segment(
                    get_box((2, 2, 2, 2))
                    + get_box((0, 2, 0, 2)) * self._plot_size.width
                    + get_box((2, 0, 2, 2))
                ),
                Segment(" " * (self._margin_right - 1), Style(bgcolor="green")),
            ]
        )

    def _render_bottom_margin(self, idx: int) -> Strip:
        if idx == 0:
            tick_labels = " " * self.size.width
            tick_labels = insert_text_into(
                tick_labels,
                str(self._x_min),
                index=self._margin_left - 1,
                align=TextAlign.CENTER,
            )
            tick_labels = insert_text_into(
                tick_labels,
                str(self._x_max),
                index=self.size.width - self._margin_right,
                align=TextAlign.RIGHT,
            )
            return Strip([Segment(tick_labels, Style.parse("bold white on black"))])
        else:
            return Strip(
                [Segment(" " * self.size.width, Style(bgcolor="red"))],
                cell_length=self.size.width,
            )


def insert_text_into(
    string: str, text: str, index: int, align: TextAlign = TextAlign.LEFT
) -> str:
    """Insert text into a string, overwriting the original string.

    This method will insert text into an existing string, overwriting the
    original and keeping its length, clipping the text if necessary. You can
    specify the text alignment (left, center or right).

    For example:

        >>> insert_text_into("0123456789", "---", 0)
        '---3456789'
        >>> insert_text_into("0123456789", "---", 4)
        '0123---789'
        >>> insert_text_into("0123456789", "---", 9)
        '012345678-'
        >>> insert_text_into("0123456789", "---", 20)
        '0123456789'
        >>> insert_text_into("0123456789", "---", -1)
        '--23456789'
        >>> insert_text_into("0123456789", "---", -2)
        '-123456789'
        >>> insert_text_into("0123456789", "---", -20)
        '0123456789'
        >>> insert_text_into("0123456789", "---", 9, align=TextAlign.RIGHT)
        '0123456---'

    Args:
        string: The original string.
        text: The text to insert.
        index: The position at which to insert.
        alignment: The text alignment (left, center or right).

    Returns:
        The new string.
    """
    if align == TextAlign.RIGHT:
        index -= len(text) - 1
    elif align == TextAlign.CENTER:
        div, mod = divmod(len(text), 2)
        index -= div
        if mod == 0:
            # even number of characters, shift one to the right since I just
            # like that better -- DF
            index += 1

    if index >= len(string) or index <= -len(text):
        return string
    elif index < 0:
        return text[-index:] + string[len(text) + index :]
    else:
        return (
            string[:index] + text[: len(string) - index] + string[index + len(text) :]
        )


class DemoApp(App[None]):
    CSS = """
        PlotWidget {
            border: solid $secondary;
        }
    """
    _phi: float = 0.0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self.set_interval(1 / 24, self.plot_refresh)

    def plot_refresh(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        plot.scatter(
            x=[0, 1, 2, 3, 4, 5, 9.99],
            y=[0, 1, 4, 9, 16, 25, 29.99],
            marker_style="blue",
            marker="*",
        )
        x = np.linspace(0, 10, 17)
        plot.plot(x=x, y=10 + 10 * np.sin(x + self._phi), line_style="red3")
        plot.plot(x=x, y=10 + 10 * np.sin(x + self._phi + 1), line_style="green")

        plot.refresh()
        self._phi += 0.1


if __name__ == "__main__":
    app = DemoApp()
    app.run()
