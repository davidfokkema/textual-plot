import enum
from dataclasses import dataclass
from math import ceil, floor
from typing import Iterator, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.segment import Segment
from rich.style import Style
from textual import on
from textual._box_drawing import BOX_CHARACTERS
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Grid, Horizontal
from textual.events import MouseScrollDown, MouseScrollUp
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Placeholder

from textual_plot.canvas import Canvas

ZOOM_FACTOR = 0.01


@dataclass
class DataSet:
    x: NDArray[np.floating]
    y: NDArray[np.floating]


@dataclass
class LinePlot(DataSet):
    line_style: str


@dataclass
class ScatterPlot(DataSet):
    marker: str
    marker_style: str


class TextAlign(enum.Enum):
    LEFT = enum.auto()
    CENTER = enum.auto()
    RIGHT = enum.auto()


class PlotWidget(Widget):
    DEFAULT_CSS = """
        PlotWidget {
            Grid {
                grid-size: 2 2;

                #bottom-margin {
                    column-span: 2;
                }
            }
        }
    """

    _datasets: list[DataSet]

    _x_min: float = 0.0
    _x_max: float = 10.0
    _y_min: float = 0.0
    _y_max: float = 30.0
    _margin_bottom: int = 0
    _margin_left: int = 0

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._datasets = []

    def compose(self) -> ComposeResult:
        with Grid():
            yield Canvas(id="left-margin")
            yield Canvas(id="plot")
            yield Canvas(id="bottom-margin")

    def on_mount(self) -> None:
        self._update_margin_sizes()

    def _on_resize(self) -> None:
        self.call_later(self.refresh)

    def _on_canvas_resize(self, event: Canvas.Resize) -> None:
        event.canvas.reset(size=event.size)

    def _update_margin_sizes(self) -> None:
        grid = self.query_one(Grid)
        grid.styles.grid_columns = f"{self._margin_left} 1fr"
        grid.styles.grid_rows = f"1fr {self._margin_bottom}"

    def clear(self) -> None:
        self._datasets = []
        self.query_one("#plot", Canvas).reset()

    def plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        line_style: str = "white",
    ) -> None:
        self._datasets.append(
            LinePlot(x=np.array(x), y=np.array(y), line_style=line_style)
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
                marker_style=marker_style,
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

    def _render_plot(self) -> None:
        if (canvas := self.query_one("#plot", Canvas))._canvas_size is None:
            return

        for dataset in self._datasets:
            if isinstance(dataset, ScatterPlot):
                self._render_scatter_plot(dataset)
            elif isinstance(dataset, LinePlot):
                self._render_line_plot(dataset)

        canvas.draw_rectangle_box(
            0, 0, canvas.size.width - 1, canvas.size.height - 1, thickness=2
        )

    def _render_scatter_plot(self, dataset: ScatterPlot) -> None:
        canvas = self.query_one("#plot", Canvas)
        pixels = [
            map_coordinate_to_pixel(
                xi,
                yi,
                self._x_min,
                self._x_max,
                self._y_min,
                self._y_max,
                Region(
                    1,
                    1,
                    canvas.size.width - 2,
                    canvas.size.height - 2,
                ),
            )
            for xi, yi in zip(dataset.x, dataset.y)
        ]

        for pixel in pixels:
            canvas.set_pixel(*pixel, char=dataset.marker, style=dataset.marker_style)

    def _render_line_plot(self, dataset: LinePlot) -> None:
        canvas = self.query_one("#plot", Canvas)
        pixels = [
            map_coordinate_to_pixel(
                xi,
                yi,
                self._x_min,
                self._x_max,
                self._y_min,
                self._y_max,
                Region(
                    1,
                    1,
                    canvas.size.width - 2,
                    canvas.size.height - 2,
                ),
            )
            for xi, yi in zip(dataset.x, dataset.y)
        ]

        for i in range(1, len(pixels)):
            canvas.draw_line(*pixels[i - 1], *pixels[i], style=dataset.line_style)

    @on(MouseScrollDown)
    def zoom_in(self, event: MouseScrollDown) -> None:
        if (offset := event.get_content_offset(self)) is not None:
            x, y = self._content_offset_to_plot_coordinate(offset)
            self._x_min = (self._x_min + ZOOM_FACTOR * x) / (1 + ZOOM_FACTOR)
            self._x_max = (self._x_max - ZOOM_FACTOR * x) / (1 + ZOOM_FACTOR)
            self._y_min = (self._y_min + ZOOM_FACTOR * y) / (1 + ZOOM_FACTOR)
            self._y_max = (self._y_max - ZOOM_FACTOR * y) / (1 + ZOOM_FACTOR)
        self.refresh()

    @on(MouseScrollUp)
    def zoom_out(self, event: MouseScrollDown) -> None:
        if (offset := event.get_content_offset(self)) is not None:
            x, y = self._content_offset_to_plot_coordinate(offset)
            self._x_min = (self._x_min - ZOOM_FACTOR * x) / (1 - ZOOM_FACTOR)
            self._x_max = (self._x_max + ZOOM_FACTOR * x) / (1 - ZOOM_FACTOR)
            self._y_min = (self._y_min - ZOOM_FACTOR * y) / (1 - ZOOM_FACTOR)
            self._y_max = (self._y_max + ZOOM_FACTOR * y) / (1 - ZOOM_FACTOR)
        self.refresh()

    def _content_offset_to_plot_coordinate(self, offset):
        x = offset.x - self._margin_left
        y = self._plot_size.height - (offset.y - self._margin_top)
        xp = self._linear_mapper(x, 0, self._plot_size.width, self._x_min, self._x_max)
        yp = self._linear_mapper(y, 0, self._plot_size.height, self._y_min, self._y_max)
        return xp, yp


def map_coordinate_to_pixel(
    x: float,
    y: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    region: Region,
) -> tuple[int, int]:
    x = floor(linear_mapper(x, xmin, xmax, region.x, region.right))
    # positive y direction is reversed
    y = ceil(linear_mapper(y, ymin, ymax, region.bottom - 1, region.y - 1))
    return x, y


def map_pixel_to_coordinate(
    px: int,
    py: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    region: Region,
) -> tuple[float, float]:
    x = linear_mapper(px + 0.5, region.x, region.right, xmin, xmax)
    # positive y direction is reversed
    y = linear_mapper(py + 0.5, region.bottom, region.y, ymin, ymax)
    return x, y


def linear_mapper(
    x: float | int,
    a: float | int,
    b: float | int,
    a_prime: float | int,
    b_prime: float | int,
) -> float:
    return a_prime + (x - a) * (b_prime - a_prime) / (b - a)


class DemoApp(App[None]):
    _phi: float = 0.0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self.set_interval(1 / 24, self.plot_refresh)
        # self.call_after_refresh(self.plot_refresh)

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

        self._phi += 0.1


if __name__ == "__main__":
    app = DemoApp()
    app.run()
