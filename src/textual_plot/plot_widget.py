from __future__ import annotations

import sys
from dataclasses import dataclass
from math import ceil, floor, log10
from typing import Iterable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from textual import on
from textual._box_drawing import BOX_CHARACTERS, combine_quads
from textual.app import ComposeResult
from textual.containers import Grid
from textual.events import MouseMove, MouseScrollDown, MouseScrollUp
from textual.geometry import Region
from textual.message import Message
from textual.widget import Widget
from textual_hires_canvas import Canvas, HiResMode, TextAlign

ZOOM_FACTOR = 0.05


@dataclass
class DataSet:
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    hires_mode: HiResMode | None


@dataclass
class LinePlot(DataSet):
    line_style: str


@dataclass
class ScatterPlot(DataSet):
    marker: str | None
    marker_style: str


class PlotWidget(Widget, can_focus=True):
    """A plot widget for Textual apps.

    This widget supports high-resolution line and scatter plots, has nice ticks
    at 1, 2, 5, 10, 20, 50, etc. intervals and supports zooming and panning with
    your pointer device.
    """

    @dataclass
    class ScaleChanged(Message):
        plot: "PlotWidget"
        x_min: float
        x_max: float
        y_min: float
        y_max: float

    DEFAULT_CSS = """
        PlotWidget {
            Grid {
                grid-size: 2 3;

                #top-margin, #bottom-margin {
                    column-span: 2;
                }
            }
        }
    """

    BINDINGS = [("r", "reset_scales", "Reset scales")]

    _datasets: list[DataSet]

    _user_x_min: float | None = None
    _user_x_max: float | None = None
    _user_y_min: float | None = None
    _user_y_max: float | None = None
    _auto_x_min: bool = True
    _auto_x_max: bool = True
    _auto_y_min: bool = True
    _auto_y_max: bool = True
    _x_min: float = 0.0
    _x_max: float = 1.0
    _y_min: float = 0.0
    _y_max: float = 1.0

    _x_ticks: Iterable[float] | None = None
    _y_ticks: Iterable[float] | None = None

    _margin_top: int = 2
    _margin_bottom: int = 3
    _margin_left: int = 10

    _x_label: str = ""
    _y_label: str = ""

    _allow_pan_and_zoom: bool = True
    _is_dragging: bool = False
    _needs_rerender: bool = False

    def __init__(
        self,
        allow_pan_and_zoom: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        *,
        disabled: bool = False,
    ) -> None:
        """Initializes the plot widget with basic widget parameters.

        Params:
            allow_pan_and_zoom: Whether to allow panning and zooming the plot.
                Defaults to True.
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes for the widget.
            disabled: Whether the widget is disabled or not.
        """
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._datasets = []
        self._allow_pan_and_zoom = allow_pan_and_zoom

    def compose(self) -> ComposeResult:
        with Grid():
            yield Canvas(1, 1, id="top-margin")
            yield Canvas(1, 1, id="left-margin")
            yield Canvas(1, 1, id="plot")
            yield Canvas(1, 1, id="bottom-margin")

    def on_mount(self) -> None:
        self._update_margin_sizes()
        self.set_xlimits(None, None)
        self.set_ylimits(None, None)
        self.clear()

    def _on_canvas_resize(self, event: Canvas.Resize) -> None:
        event.canvas.reset(size=event.size)
        self._needs_rerender = True
        self.call_later(self.refresh)

    def _update_margin_sizes(self) -> None:
        grid = self.query_one(Grid)
        grid.styles.grid_columns = f"{self._margin_left} 1fr"
        grid.styles.grid_rows = f"{self._margin_top} 1fr {self._margin_bottom}"

    def clear(self) -> None:
        """Clear the plot canvas."""
        self._datasets = []
        self._needs_rerender = True
        self.refresh()

    def plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        line_style: str = "white",
        hires_mode: HiResMode | None = None,
    ) -> None:
        """Graph dataset using a line plot.

        If you supply hires_mode, the dataset will be plotted using one of the
        available high-resolution modes like 1x2, 2x2 or 2x8 pixel-per-cell
        characters.

        Args:
            x: An ArrayLike with the data values for the horizontal axis.
            y: An ArrayLike with the data values for the vertical axis.
            line_style: A string with the style of the line. Defaults to
                "white".
            hires_mode: A HiResMode enum or None to plot with full-height
                blocks. Defaults to None.
        """
        x, y = drop_nans_and_infs(np.array(x), np.array(y))
        self._datasets.append(
            LinePlot(
                x=x,
                y=y,
                line_style=line_style,
                hires_mode=hires_mode,
            )
        )
        self._needs_rerender = True
        self.refresh()

    def scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        marker: str = "o",
        marker_style: str = "white",
        hires_mode: HiResMode | None = None,
    ) -> None:
        """Graph dataset using a scatter plot.

        If you supply hires_mode, the dataset will be plotted using one of the
        available high-resolution modes like 1x2, 2x2 or 2x8 pixel-per-cell
        characters.

        Args:
            x: An ArrayLike with the data values for the horizontal axis.
            y: An ArrayLike with the data values for the vertical axis.
            marker: A string with the character to print as the marker.
            marker_style: A string with the style of the marker. Defaults to
                "white".
            hires_mode: A HiResMode enum or None to plot with the supplied
                marker. Defaults to None.
        """
        x, y = drop_nans_and_infs(np.array(x), np.array(y))
        self._datasets.append(
            ScatterPlot(
                x=x,
                y=y,
                marker=marker,
                marker_style=marker_style,
                hires_mode=hires_mode,
            )
        )
        self._needs_rerender = True
        self.refresh()

    def set_xlimits(self, xmin: float | None = None, xmax: float | None = None) -> None:
        """Set the limits of the x axis.

        Args:
            xmin: A float with the minimum x value or None for autoscaling.
                Defaults to None.
            xmax: A float with the maximum x value or None for autoscaling.
                Defaults to None.
        """
        self._user_x_min = xmin
        self._user_x_max = xmax
        self._auto_x_min = xmin is None
        self._auto_x_max = xmax is None
        self._x_min = xmin if xmin is not None else 0.0
        self._x_max = xmax if xmax is not None else 1.0
        self._needs_rerender = True
        self.refresh()

    def set_ylimits(self, ymin: float | None = None, ymax: float | None = None) -> None:
        """Set the limits of the y axis.

        Args:
            xmin: A float with the minimum y value or None for autoscaling.
                Defaults to None.
            xmax: A float with the maximum y value or None for autoscaling.
                Defaults to None.
        """
        self._user_y_min = ymin
        self._user_y_max = ymax
        self._auto_y_min = ymin is None
        self._auto_y_max = ymax is None
        self._y_min = ymin if ymin is not None else 0.0
        self._y_max = ymax if ymax is not None else 1.0
        self._needs_rerender = True
        self.refresh()

    def set_xlabel(self, label: str) -> None:
        """Set a label for the x axis.

        Args:
            label: A string with the label text.
        """
        self._x_label = label

    def set_ylabel(self, label: str) -> None:
        """Set a label for the y axis.

        Args:
            label: A string with the label text.
        """
        self._y_label = label

    def set_xticks(self, ticks: Iterable[float] | None = None) -> None:
        """Set the x axis ticks.

        Use None for autoscaling, an empty list to hide the ticks.

        Args:
            ticks: An iterable with the tick values.
        """
        self._x_ticks = ticks

    def set_yticks(self, ticks: Iterable[float] | None = None) -> None:
        """Set the y axis ticks.

        Use None for autoscaling, an empty list to hide the ticks.

        Args:
            ticks: An iterable with the tick values.
        """
        self._y_ticks = ticks

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        """Refresh the widget."""
        self._render_plot()
        return super().refresh(
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    def _render_plot(self) -> None:
        if (canvas := self.query_one("#plot", Canvas))._canvas_size is None:
            return

        if self._needs_rerender:
            self._needs_rerender = False
            # clear canvas
            canvas.reset()

            # determine axis limits
            if self._datasets:
                xs = [dataset.x for dataset in self._datasets]
                ys = [dataset.y for dataset in self._datasets]
                if self._auto_x_min:
                    self._x_min = min(np.min(x) for x in xs)
                if self._auto_x_max:
                    self._x_max = max(np.max(x) for x in xs)
                if self._auto_y_min:
                    self._y_min = min(np.min(y) for y in ys)
                if self._auto_y_max:
                    self._y_max = max(np.max(y) for y in ys)

                if self._x_min == self._x_max:
                    self._x_min -= 1e-6
                    self._x_max += 1e-6
                if self._y_min == self._y_max:
                    self._y_min -= 1e-6
                    self._y_max += 1e-6

            # render datasets
            for dataset in self._datasets:
                if isinstance(dataset, ScatterPlot):
                    self._render_scatter_plot(dataset)
                elif isinstance(dataset, LinePlot):
                    self._render_line_plot(dataset)

            # render axis, ticks and labels
            canvas.draw_rectangle_box(
                0, 0, canvas.size.width - 1, canvas.size.height - 1, thickness=2
            )
            self._render_x_ticks()
            self._render_y_ticks()
            self._render_x_label()
            self._render_y_label()

    def _render_scatter_plot(self, dataset: ScatterPlot) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None
        if dataset.hires_mode:
            pixels = [
                self.get_hires_pixel_from_coordinate(xi, yi)
                for xi, yi in zip(dataset.x, dataset.y)
            ]
            canvas.set_hires_pixels(
                pixels, style=dataset.marker_style, hires_mode=dataset.hires_mode
            )
        else:
            pixels = [
                self.get_pixel_from_coordinate(xi, yi)
                for xi, yi in zip(dataset.x, dataset.y)
            ]
            for pixel in pixels:
                assert dataset.marker is not None
                canvas.set_pixel(
                    *pixel, char=dataset.marker, style=dataset.marker_style
                )

    def _render_line_plot(self, dataset: LinePlot) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None

        if dataset.hires_mode:
            pixels = [
                self.get_hires_pixel_from_coordinate(xi, yi)
                for xi, yi in zip(dataset.x, dataset.y)
            ]
            coordinates = [(*pixels[i - 1], *pixels[i]) for i in range(1, len(pixels))]
            canvas.draw_hires_lines(
                coordinates, style=dataset.line_style, hires_mode=dataset.hires_mode
            )
        else:
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

        if self._x_ticks is None:
            x_ticks, x_labels = self.get_ticks_between(self._x_min, self._x_max)
        else:
            x_ticks = self._x_ticks
            x_labels = self.get_labels_for_ticks(x_ticks)
        for tick, label in zip(x_ticks, x_labels):
            if tick < self._x_min or tick > self._x_max:
                continue
            align = TextAlign.CENTER
            # only interested in the x-coordinate, set y to 0.0
            x, _ = self.get_pixel_from_coordinate(tick, 0.0)
            if tick == self._x_min:
                x -= 1
            elif tick == self._x_max:
                align = TextAlign.RIGHT
            for y, quad in [
                # put ticks at top and bottom of scale rectangle
                (0, (2, 0, 0, 0)),
                (canvas.scale_rectangle.bottom, (0, 0, 2, 0)),
            ]:
                new_pixel = self.combine_quad_with_pixel(quad, canvas, x, y)
                canvas.set_pixel(x, y, new_pixel)
            bottom_margin.write_text(x + self._margin_left, 0, label, align)

    def _render_y_ticks(self) -> None:
        canvas = self.query_one("#plot", Canvas)
        assert canvas.scale_rectangle is not None
        left_margin = self.query_one("#left-margin", Canvas)
        left_margin.reset()

        if self._y_ticks is None:
            y_ticks, y_labels = self.get_ticks_between(self._y_min, self._y_max)
        else:
            y_ticks = self._y_ticks
            y_labels = self.get_labels_for_ticks(y_ticks)
        align = TextAlign.RIGHT
        for tick, label in zip(y_ticks, y_labels):
            if tick < self._y_min or tick > self._y_max:
                continue
            # only interested in the x-coordinate, set x to 0.0
            _, y = self.get_pixel_from_coordinate(0.0, tick)
            if tick == self._y_min:
                y += 1
            for x, quad in [
                # put ticks at left and right of scale rectangle
                (0, (0, 0, 0, 2)),
                (canvas.scale_rectangle.right, (0, 2, 0, 0)),
            ]:
                new_pixel = self.combine_quad_with_pixel(quad, canvas, x, y)
                canvas.set_pixel(x, y, new_pixel)
            left_margin.write_text(self._margin_left - 2, y, label, align)

    def _render_x_label(self) -> None:
        canvas = self.query_one("#plot", Canvas)
        margin = self.query_one("#bottom-margin", Canvas)
        margin.write_text(
            canvas.size.width // 2 + self._margin_left,
            2,
            self._x_label,
            TextAlign.CENTER,
        )

    def _render_y_label(self) -> None:
        margin = self.query_one("#top-margin", Canvas)
        margin.write_text(
            self._margin_left - 2,
            0,
            self._y_label,
            TextAlign.CENTER,
        )

    def get_ticks_between(
        self, min_: float, max_: float, max_ticks: int = 8
    ) -> tuple[list[float], list[str]]:
        delta_x = max_ - min_
        tick_spacing = delta_x / 5
        power = floor(log10(tick_spacing))
        approx_interval = tick_spacing / 10**power
        intervals = np.array([1, 2, 5, 10])

        idx = intervals.searchsorted(approx_interval)
        interval = (intervals[idx - 1] if idx > 0 else intervals[0]) * 10**power
        if delta_x // interval > max_ticks:
            interval = intervals[idx] * 10**power
        ticks = [
            float(t * interval)
            for t in np.arange(ceil(min_ / interval), max_ // interval + 1)
        ]
        decimals = -min(0, power)
        tick_labels = self.get_labels_for_ticks(ticks, decimals)
        return ticks, tick_labels

    def get_labels_for_ticks(
        self, ticks: list[float], decimals: int | None = None
    ) -> list[str]:
        """Generate formatted labels for given tick values.

        Args:
            ticks: A list of tick values to be formatted.
            decimals: The number of decimal places for formatting the tick values.

        Returns:
            A list of formatted tick labels as strings.
        """
        if not ticks:
            return []
        if decimals is None:
            if len(ticks) >= 2:
                power = floor(log10(ticks[1] - ticks[0]))
            else:
                power = 0
            decimals = -min(0, power)
        tick_labels = [f"{tick:.{decimals}f}" for tick in ticks]
        return tick_labels

    def combine_quad_with_pixel(
        self, quad: tuple[int, int, int, int], canvas: Canvas, x: int, y: int
    ) -> str:
        pixel = canvas.get_pixel(x, y)[0]
        for current_quad, v in BOX_CHARACTERS.items():
            if v == pixel:
                break
        new_quad = combine_quads(current_quad, quad)
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

    def get_hires_pixel_from_coordinate(
        self, x: float | np.floating, y: float | np.floating
    ) -> tuple[float | np.floating, float | np.floating]:
        assert (
            scale_rectangle := self.query_one("#plot", Canvas).scale_rectangle
        ) is not None
        return map_coordinate_to_hires_pixel(
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

    def _zoom(self, event: MouseScrollDown | MouseScrollUp, factor: float) -> None:
        if not self._allow_pan_and_zoom:
            return
        if (offset := event.get_content_offset(self)) is not None:
            widget, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
            canvas = self.query_one("#plot", Canvas)
            assert canvas.scale_rectangle is not None
            if widget.id == "bottom-margin":
                offset = event.screen_offset - self.screen.get_offset(canvas)
            x, y = self.get_coordinate_from_pixel(offset.x, offset.y)
            if widget.id in ("plot", "bottom-margin"):
                self._auto_x_min = False
                self._auto_x_max = False
                self._x_min = (self._x_min + factor * x) / (1 + factor)
                self._x_max = (self._x_max + factor * x) / (1 + factor)
            if widget.id in ("plot", "left-margin"):
                self._auto_y_min = False
                self._auto_y_max = False
                self._y_min = (self._y_min + factor * y) / (1 + factor)
                self._y_max = (self._y_max + factor * y) / (1 + factor)
            self.post_message(
                self.ScaleChanged(
                    self, self._x_min, self._x_max, self._y_min, self._y_max
                )
            )
            self._needs_rerender = True
            self.call_later(self.refresh)

    @on(MouseScrollDown)
    def zoom_in(self, event: MouseScrollDown) -> None:
        event.stop()
        self._zoom(event, ZOOM_FACTOR)

    @on(MouseScrollUp)
    def zoom_out(self, event: MouseScrollUp) -> None:
        event.stop()
        self._zoom(event, -ZOOM_FACTOR)

    @on(MouseMove)
    def pan_plot(self, event: MouseMove) -> None:
        if not self._allow_pan_and_zoom:
            return
        if event.button == 0:
            # If no button is pressed, don't drag.
            return

        x1, y1 = self.get_coordinate_from_pixel(1, 1)
        x2, y2 = self.get_coordinate_from_pixel(2, 2)
        dx, dy = x2 - x1, y1 - y2

        assert event.widget is not None
        if event.widget.id in ("plot", "bottom-margin"):
            self._auto_x_min = False
            self._auto_x_max = False
            self._x_min -= dx * event.delta_x
            self._x_max -= dx * event.delta_x
        if event.widget.id in ("plot", "left-margin"):
            self._auto_y_min = False
            self._auto_y_max = False
            self._y_min += dy * event.delta_y
            self._y_max += dy * event.delta_y
        self.post_message(
            self.ScaleChanged(self, self._x_min, self._x_max, self._y_min, self._y_max)
        )
        self._needs_rerender = True
        self.call_later(self.refresh)

    def action_reset_scales(self) -> None:
        self.set_xlimits(self._user_x_min, self._user_x_max)
        self.set_ylimits(self._user_y_min, self._user_y_max)
        self.post_message(
            self.ScaleChanged(self, self._x_min, self._x_max, self._y_min, self._y_max)
        )
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


def map_coordinate_to_hires_pixel(
    x: float | np.floating,
    y: float | np.floating,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    region: Region,
) -> tuple[float | np.floating, float | np.floating]:
    x = linear_mapper(x, xmin, xmax, region.x, region.right)
    # positive y direction is reversed
    y = linear_mapper(y, ymin, ymax, region.bottom, region.y)
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


def drop_nans_and_infs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaNs and Infs from x and y arrays.

    Args:
        x: An array with the data values for the horizontal axis.
        y: An array with the data values for the vertical axis.

    Returns:
        A tuple of arrays with NaNs and Infs removed.
    """
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    return x[mask], y[mask]
