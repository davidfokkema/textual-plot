import random
import time
from dataclasses import dataclass
from typing import Self

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


class LinePlot(DataSet): ...


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

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.clear()

    def clear(self) -> None:
        self._datasets = []

    def plot(self, x: ArrayLike, y: ArrayLike) -> None:
        self._datasets.append(LinePlot(x=x, y=y))
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

    def _render_plot(self) -> None:
        self._plot_size = self.size
        if self._plot_size.width == 0:
            return

        self._canvas = [
            [Segment(" ") for _ in range(self._plot_size.width)]
            for _ in range(self._plot_size.height)
        ]
        # x_step = (x_max - x_min) / plot_size.width
        for dataset in self._datasets:
            if isinstance(dataset, ScatterPlot):
                self._render_scatter_plot(dataset)
            # x = np.arange(x_min + x_step / 2, x_max, x_step)
            # assert len(x) == plot_size.width

    def _render_scatter_plot(self, dataset: ScatterPlot) -> None:
        x_cell_boundaries = np.linspace(
            self._x_min, self._x_max, self._plot_size.width + 1
        )
        y_cell_boundaries = np.linspace(
            self._y_min, self._y_max, self._plot_size.height + 1
        )

        x_indexes = np.searchsorted(x_cell_boundaries, dataset.x)
        y_indexes = np.searchsorted(y_cell_boundaries, dataset.y)
        for x, y in zip(x_indexes, y_indexes):
            try:
                self._canvas[y][x] = Segment(
                    text=dataset.marker, style=dataset.marker_style
                )
            except IndexError:
                # data point is outside plot area
                pass

    def render_line(self, y: int) -> Strip:
        """Render a line of the widget. y is relative to the top of the widget."""
        return Strip(
            self._canvas[self._plot_size.height - y - 1],
            cell_length=self._plot_size.width,
        )


class DemoApp(App[None]):
    _phi = 0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self.set_interval(1 / 60, self.plot_refresh)

    def plot_refresh(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        plot.scatter(x=[0, 1, 2, 3, 4, 5], y=[0, 1, 4, 9, 16, 25], marker_style="blue")
        x = np.linspace(0, 10, 1001)
        plot.scatter(
            x=x, y=10 + 10 * np.sin(x + self._phi), marker="█", marker_style="red3"
        )
        plot.refresh()
        self._phi += 0.1


if __name__ == "__main__":
    app = DemoApp()
    app.run()
