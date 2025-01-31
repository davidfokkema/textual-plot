from dataclasses import dataclass
from math import ceil, floor
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from textual import on
from textual._box_drawing import BOX_CHARACTERS, combine_quads
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.events import MouseMove, MouseScrollDown, MouseScrollUp
from textual.geometry import Region
from textual.widget import Widget

from textual_plot.canvas import Canvas, TextAlign

ZOOM_FACTOR = 0.05


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
    _margin_bottom: int = 3
    _margin_left: int = 10

    _is_dragging: bool = False

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
        # clear canvas
        canvas.reset()

        for dataset in self._datasets:
            if isinstance(dataset, ScatterPlot):
                self._render_scatter_plot(dataset)
            elif isinstance(dataset, LinePlot):
                self._render_line_plot(dataset)

        canvas.draw_rectangle_box(
            0, 0, canvas.size.width - 1, canvas.size.height - 1, thickness=2
        )
        self._render_x_ticks()
        self._render_y_ticks()

    def _render_scatter_plot(self, dataset: ScatterPlot) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None
        pixels = [
            self.get_pixel_from_coordinate(xi, yi)
            for xi, yi in zip(dataset.x, dataset.y)
        ]

        for pixel in pixels:
            canvas.set_pixel(*pixel, char=dataset.marker, style=dataset.marker_style)

    def _render_line_plot(self, dataset: LinePlot) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None
        pixels = [
            self.get_pixel_from_coordinate(xi, yi)
            for xi, yi in zip(dataset.x, dataset.y)
        ]

        for i in range(1, len(pixels)):
            canvas.draw_line(*pixels[i - 1], *pixels[i], style=dataset.line_style)

    def _render_x_ticks(self) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None
        bottom_margin = self.query_one("#bottom-margin", Canvas)
        bottom_margin.reset()

        x_ticks = np.linspace(self._x_min, self._x_max, 5)
        for tick in x_ticks:
            align = TextAlign.CENTER
            # only interested in the x-coordinate, set y to 0.0
            x, _ = self.get_pixel_from_coordinate(tick, 0.0)
            if tick == self._x_min:
                x -= 1
            elif tick == self._x_max:
                align = TextAlign.RIGHT
            for y, quad in [
                (0, (2, 0, 0, 0)),
                (canvas.scale_rectangle.bottom, (0, 0, 2, 0)),
            ]:
                new_pixel = self.combine_quad_with_pixel(quad, canvas, x, y)
                canvas.set_pixel(x, y, new_pixel)
            bottom_margin.write_text(x + self._margin_left, 0, f"{tick:.1f}", align)

    def _render_y_ticks(self) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None
        left_margin = self.query_one("#left-margin", Canvas)
        left_margin.reset()

        y_ticks = np.linspace(self._y_min, self._y_max, 5)
        align = TextAlign.RIGHT
        for tick in y_ticks:
            # only interested in the x-coordinate, set x to 0.0
            _, y = self.get_pixel_from_coordinate(0.0, tick)
            if tick == self._y_min:
                y += 1
            for x, quad in [
                (0, (0, 0, 0, 2)),
                (canvas.scale_rectangle.right, (0, 2, 0, 0)),
            ]:
                new_pixel = self.combine_quad_with_pixel(quad, canvas, x, y)
                canvas.set_pixel(x, y, new_pixel)
            left_margin.write_text(self._margin_left - 2, y, f"{tick:.1f}", align)

    def combine_quad_with_pixel(
        self, quad: tuple[int, int, int, int], canvas: Canvas, x: int, y: int
    ) -> str:
        pixel = canvas.get_pixel(x, y)[0]
        for current_quad, v in BOX_CHARACTERS.items():
            if v == pixel:
                break
        new_quad = combine_quads(current_quad, quad)
        print(f"{current_quad=}, {quad=}, {new_quad=}")
        return BOX_CHARACTERS[new_quad]

    def get_pixel_from_coordinate(
        self, x: float | np.floating, y: float | np.floating
    ) -> tuple[int, int]:
        assert (
            scale_rectangle := self.query_one("#plot", Canvas).scale_rectangle
        ) is not None
        return map_coordinate_to_pixel(
            x,
            y,
            self._x_min,
            self._x_max,
            self._y_min,
            self._y_max,
            region=scale_rectangle,
        )

    def get_coordinate_from_pixel(self, x: int, y: int) -> tuple[float, float]:
        assert (
            scale_rectangle := self.query_one("#plot", Canvas).scale_rectangle
        ) is not None
        return map_pixel_to_coordinate(
            x,
            y,
            self._x_min,
            self._x_max,
            self._y_min,
            self._y_max,
            region=scale_rectangle,
        )

    @on(MouseScrollDown)
    def zoom_in(self, event: MouseScrollDown) -> None:
        if (offset := event.get_content_offset(self)) is not None:
            widget, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
            canvas = self.query_one("#plot", Canvas)
            assert canvas.scale_rectangle is not None
            if widget.id == "bottom-margin":
                # Bottom margin is wider than the plot canvas. For x-scale
                # zooming, we want to have the x-coordinate relative to the plot
                # canvas, not relative to the bottom margin.
                offset = event.screen_offset - self.screen.get_offset(canvas)
            x, y = self.get_coordinate_from_pixel(offset.x, offset.y)
            if widget.id in ("plot", "bottom-margin"):
                self._x_min = (self._x_min + ZOOM_FACTOR * x) / (1 + ZOOM_FACTOR)
                self._x_max = (self._x_max + ZOOM_FACTOR * x) / (1 + ZOOM_FACTOR)
            if widget.id in ("plot", "left-margin"):
                self._y_min = (self._y_min + ZOOM_FACTOR * y) / (1 + ZOOM_FACTOR)
                self._y_max = (self._y_max + ZOOM_FACTOR * y) / (1 + ZOOM_FACTOR)
            self.refresh()

    @on(MouseScrollUp)
    def zoom_out(self, event: MouseScrollDown) -> None:
        if (offset := event.get_content_offset(self)) is not None:
            widget, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
            canvas = self.query_one("#plot", Canvas)
            assert canvas.scale_rectangle is not None
            if widget.id == "bottom-margin":
                # Bottom margin is wider than the plot canvas. For x-scale
                # zooming, we want to have the x-coordinate relative to the plot
                # canvas, not relative to the bottom margin.
                offset = event.screen_offset - self.screen.get_offset(canvas)
            x, y = self.get_coordinate_from_pixel(offset.x, offset.y)
            if widget.id in ("plot", "bottom-margin"):
                self._x_min = (self._x_min - ZOOM_FACTOR * x) / (1 - ZOOM_FACTOR)
                self._x_max = (self._x_max - ZOOM_FACTOR * x) / (1 - ZOOM_FACTOR)
            if widget.id in ("plot", "left-margin"):
                self._y_min = (self._y_min - ZOOM_FACTOR * y) / (1 - ZOOM_FACTOR)
                self._y_max = (self._y_max - ZOOM_FACTOR * y) / (1 - ZOOM_FACTOR)
            self.refresh()

    @on(MouseMove)
    def drag_plot(self, event: MouseMove) -> None:
        if event.button == 0:
            # If no button is pressed, don't drag.
            return
        if event.screen_offset == (0, 0) or (
            event.x == event.delta_x or event.y == event.delta_y
        ):
            # FIXME: BUG in Textual?
            # If the mouse moves out of the window, the position jumps to (0, 0)
            # briefly, creating large offsets equal to the current position in
            # the next move event.
            return
        x1, y1 = self.get_coordinate_from_pixel(1, 1)
        x2, y2 = self.get_coordinate_from_pixel(2, 2)
        dx, dy = x2 - x1, y1 - y2

        assert event.widget is not None
        if event.widget.id in ("plot", "bottom-margin"):
            self._x_min -= dx * event.delta_x
            self._x_max -= dx * event.delta_x
        if event.widget.id in ("plot", "left-margin"):
            self._y_min += dy * event.delta_y
            self._y_max += dy * event.delta_y
        self.refresh()


def map_coordinate_to_pixel(
    x: float | np.floating,
    y: float | np.floating,
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
    return float(x), float(y)


def linear_mapper(
    x: float | np.floating | int,
    a: float | int,
    b: float | int,
    a_prime: float | int,
    b_prime: float | int,
) -> float | np.floating:
    return a_prime + (x - a) * (b_prime - a_prime) / (b - a)


class DemoApp(App[None]):
    _phi: float = 0.0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        # self.set_interval(1 / 24, self.plot_refresh)
        self.plot_refresh()

    def plot_refresh(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        x = np.linspace(0, 10, 41)
        y = x**2 / 3.5
        plot.scatter(
            x,
            y,
            marker_style="blue",
            marker="*",
        )
        x = np.linspace(0, 10, 200)
        plot.plot(x=x, y=10 + 10 * np.sin(x + self._phi), line_style="red3")
        plot.plot(x=x, y=10 + 10 * np.sin(x + self._phi + 1), line_style="green")

        self._phi += 0.1


if __name__ == "__main__":
    app = DemoApp()
    app.run()
