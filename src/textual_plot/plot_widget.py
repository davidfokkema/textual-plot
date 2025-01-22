import random
import time
from dataclasses import dataclass
from typing import Iterator, Self

import numpy as np
from numpy.typing import ArrayLike
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.color import Color
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widget import Widget


@dataclass
class DataSet:
    x: np.ndarray
    y: np.ndarray


@dataclass
class LinePlot(DataSet):
    line_style: Style


@dataclass
class ScatterPlot(DataSet):
    marker: str
    marker_style: Style


class PlotWidget(Widget):
    _datasets: list[DataSet]

    _x_min: float = 0.0
    _x_max: float = 10.0
    _y_min: float = 0.0
    _y_max: float = 30.0
    _margin_left: int = 10
    _margin_bottom: int = 3

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.clear()

    def clear(self) -> None:
        self._datasets = []

    def plot(self, x: ArrayLike, y: ArrayLike, line_style="white") -> None:
        self._datasets.append(LinePlot(x=x, y=y, line_style=Style.parse(line_style)))
        self.refresh()

    def scatter(
        self, x: ArrayLike, y: ArrayLike, marker="o", marker_style="white"
    ) -> None:
        self._datasets.append(
            ScatterPlot(x=x, y=y, marker=marker, marker_style=Style.parse(marker_style))
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

    def _linear_mapper(self, x, a, b, a_prime, b_prime):
        return round(a_prime + (x - a) * (b_prime - a_prime) / (b - a))

    def _map_coordinate_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (
            self._linear_mapper(x, self._x_min, self._x_max, 0, self.size.width - 1),
            self._linear_mapper(y, self._y_min, self._y_max, 0, self.size.height - 1),
        )

    def _map_coordinates_to_pixels(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[list[float], list[float]]:
        return (
            [
                self._linear_mapper(
                    px, self._x_min, self._x_max, 0, self.size.width - 1
                )
                for px in x
            ],
            [
                self._linear_mapper(
                    py, self._y_min, self._y_max, 0, self.size.height - 1
                )
                for py in y
            ],
        )

    def _render_plot(self) -> None:
        self._plot_size = self.size - (self._margin_left, self._margin_bottom)

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

    def bresenham_line(self, x0, y0, x1, y1):
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
        if y < self._plot_size.height:
            return Strip(
                [Segment(" " * self._margin_left, Style(bgcolor="blue"))]
                + self._canvas[self._plot_size.height - y - 1],
                cell_length=self.size.width,
            )
        else:
            return Strip(
                [Segment(" " * self.size.width, Style(bgcolor="red"))],
                cell_length=self.size.width,
            )


class DemoApp(App[None]):
    _phi = 0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self.set_interval(1 / 24, self.plot_refresh)

    def plot_refresh(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        plot.scatter(x=[0, 1, 2, 3, 4, 5], y=[0, 1, 4, 9, 16, 25], marker_style="blue")
        x = np.linspace(0, 10, 17)
        plot.plot(x=x, y=10 + 10 * np.sin(x + self._phi), line_style="red3")
        plot.plot(x=x, y=10 + 10 * np.sin(x + self._phi + 1), line_style="green")

        plot.refresh()
        self._phi += 0.1


if __name__ == "__main__":
    app = DemoApp()
    app.run()
