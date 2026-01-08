"""A Textual widget for plotting data with customizable axes, legends, and multiple plot types.

This module provides the PlotWidget class, which enables creation of interactive
plots in Textual applications. It supports scatter plots and line plots, with
features like automatic scaling, legends, high-resolution rendering and
interactive features like zooming and panning.
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from math import ceil, floor, log10
from typing import Sequence, TypeAlias

from rich.text import Text

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
import numpy as np
from numpy.typing import ArrayLike, NDArray
from textual import on
from textual._box_drawing import BOX_CHARACTERS, combine_quads
from textual.app import ComposeResult, RenderResult
from textual.containers import Grid
from textual.css.query import NoMatches
from textual.events import (
    MouseDown,
    MouseMove,
    MouseScrollDown,
    MouseScrollUp,
    MouseUp,
)
from textual.geometry import Offset, Region
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static
from textual_hires_canvas import Canvas, HiResMode, TextAlign

from textual_plot.axis_formatter import AxisFormatter, NumericAxisFormatter

__all__ = ["HiResMode", "LegendLocation", "PlotWidget"]

FloatScalar: TypeAlias = float | np.floating
FloatArray: TypeAlias = NDArray[np.floating]


ZOOM_FACTOR = 0.05

LEGEND_LINE = {
    None: "███",
    HiResMode.BRAILLE: "⠒⠒⠒",
    HiResMode.HALFBLOCK: "▀▀▀",
    HiResMode.QUADRANT: "▀▀▀",
}

LEGEND_MARKER = {
    HiResMode.BRAILLE: "⠂",
    HiResMode.HALFBLOCK: "▀",
    HiResMode.QUADRANT: "▘",
}


class LegendLocation(enum.Enum):
    """An enum to specify the location of the legend in the plot widget."""

    TOPLEFT = enum.auto()
    TOPRIGHT = enum.auto()
    BOTTOMLEFT = enum.auto()
    BOTTOMRIGHT = enum.auto()


@dataclass
class DataSet:
    """Base class for plot datasets containing coordinate data and rendering mode.

    Attributes:
        x: Array of x-coordinate values for the dataset.
        y: Array of y-coordinate values for the dataset.
        hires_mode: High-resolution rendering mode or None for standard rendering.
    """

    x: FloatArray
    y: FloatArray
    hires_mode: HiResMode | None


@dataclass
class LinePlot(DataSet):
    """A dataset for rendering as a line plot.

    Attributes:
        line_style: Rich style string for the line (e.g., "white", "bold red").
    """

    line_style: str


@dataclass
class ScatterPlot(DataSet):
    """A dataset for rendering as a scatter plot.

    Attributes:
        marker: Character to use as the marker (e.g., "o", "*", "+").
        marker_style: Rich style string for the markers (e.g., "white", "bold blue").
    """

    marker: str
    marker_style: str


@dataclass
class ErrorBarPlot(ScatterPlot):
    """A dataset for rendering as an error bar plot.

    Attributes:
        xerr: Array of x-direction error values (can be None for no x errors).
        yerr: Array of y-direction error values (can be None for no y errors).
    """

    xerr: FloatArray
    yerr: FloatArray


@dataclass
class VLinePlot:
    """A vertical line to be drawn on the plot.

    Attributes:
        x: X-coordinate where the vertical line is positioned.
        line_style: Rich style string for the line (e.g., "white", "dashed red").
    """

    x: float
    line_style: str


class Legend(Static):
    """A legend widget for the PlotWidget."""

    ALLOW_SELECT = False


class PlotWidget(Widget, can_focus=True):
    """A plot widget for Textual apps.

    This widget supports high-resolution line and scatter plots, has nice ticks
    at 1, 2, 5, 10, 20, 50, etc. intervals and supports zooming and panning with
    your pointer device.

    The following component classes can be used to style the widget:

    | Class | Description |
    | :- | :- |
    | `plot--axis` | Style of the axes (may be used to change the color). |
    | `plot--tick` | Style of the tick labels along the axes. |
    | `plot--label` | Style of axis labels. |
    """

    @dataclass
    class ScaleChanged(Message):
        """Message posted when the plot scale (axis limits) changes.

        Attributes:
            plot: The PlotWidget instance that posted the message.
            x_min: Minimum value of the x-axis after the change.
            x_max: Maximum value of the x-axis after the change.
            y_min: Minimum value of the y-axis after the change.
            y_max: Maximum value of the y-axis after the change.
        """

        plot: "PlotWidget"
        x_min: float
        x_max: float
        y_min: float
        y_max: float

    COMPONENT_CLASSES = {"plot--axis", "plot--tick", "plot--label"}

    DEFAULT_CSS = """
        PlotWidget {
            layers: plot legend;

            &:focus > .plot--axis {
                color: $primary;
            }

            & > .plot--axis {
                color: $secondary;
            }

            & > .plot--tick {
                color: $secondary;
                text-style: bold;
            }

            & > .plot--label {
                color: $primary;
                text-style: bold italic;
            }

            Grid {
                layer: plot;
                grid-size: 2 3;

                #margin-top, #margin-bottom {
                    column-span: 2;
                }
            }

            #legend {
              layer: legend;
              width: auto;
              border: solid white;
              display: none;

              &.dragged {
                border: heavy yellow;
              }
            }
        }
    """

    BINDINGS = [("r", "reset_scales", "Reset scales")]

    margin_top = reactive(2)
    margin_bottom = reactive(3)
    margin_left = reactive(10)

    _datasets: list[DataSet]
    _labels: list[str | None]

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

    _x_ticks: Sequence[float] | None = None
    _y_ticks: Sequence[float] | None = None
    _x_formatter: AxisFormatter
    _y_formatter: AxisFormatter

    _scale_rectangle: Region = Region(0, 0, 0, 0)
    _legend_location: LegendLocation = LegendLocation.TOPRIGHT
    _legend_relative_offset: Offset = Offset(0, 0)

    _x_label: str = ""
    _y_label: str = ""

    _allow_pan_and_zoom: bool = True
    _is_dragging_legend: bool = False
    _needs_rerender: bool = False

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        *,
        allow_pan_and_zoom: bool = True,
        invert_mouse_wheel: bool = False,
        disabled: bool = False,
    ) -> None:
        """Initializes the plot widget with basic widget parameters.

        Params:
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes for the widget.
            allow_pan_and_zoom: Whether to allow panning and zooming the plot.
                Defaults to True.
            invert_mouse_wheel: When set to True the zooming direction is inverted
                when scrolling in and out of the widget. Defaults to False.
            disabled: Whether the widget is disabled or not.
        """
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._datasets = []
        self._labels = []
        self._v_lines: list[VLinePlot] = []
        self._v_lines_labels: list[str | None] = []
        self._allow_pan_and_zoom = allow_pan_and_zoom
        self.invert_mouse_wheel = invert_mouse_wheel
        self._x_formatter = NumericAxisFormatter()
        self._y_formatter = NumericAxisFormatter()

    def compose(self) -> ComposeResult:
        """Compose the child widgets of the PlotWidget.

        Returns:
            An iterable of child widgets including the plot canvas, margins, and legend.
        """
        with Grid():
            yield Canvas(1, 1, id="margin-top")
            yield Canvas(1, 1, id="margin-left")
            yield Canvas(1, 1, id="plot")
            yield Canvas(1, 1, id="margin-bottom")
        yield Legend(id="legend")

    def on_mount(self) -> None:
        """Initialize the plot widget when mounted to the DOM."""
        self._update_margin_sizes()
        self.set_xlimits(None, None)
        self.set_ylimits(None, None)
        self.clear()

    def notify_style_update(self) -> None:
        """Called when styles update (e.g., theme change). Rerender the plot."""
        self.refresh(layout=True)

    def _on_canvas_resize(self, event: Canvas.Resize) -> None:
        """Handle canvas resize events to update the plot scale rectangle.

        Args:
            event: The canvas resize event containing the new size.
        """
        if event.canvas.id == "plot":
            # The scale rectangle falls just inside the axis rectangle
            self._scale_rectangle = Region(
                1, 1, event.size.width - 2, event.size.height - 2
            )
        event.canvas.reset(size=event.size)
        self._position_legend()
        self.refresh(layout=True)

    def watch_margin_top(self) -> None:
        """React to changes in the top margin reactive attribute."""
        self._update_margin_sizes()

    def watch_margin_bottom(self) -> None:
        """React to changes in the bottom margin reactive attribute."""
        self._update_margin_sizes()

    def watch_margin_left(self) -> None:
        """React to changes in the left margin reactive attribute."""
        self._update_margin_sizes()

    def _update_margin_sizes(self) -> None:
        """Update grid layout taking plot margins into account."""
        grid = self.query_one(Grid)
        grid.styles.grid_columns = f"{self.margin_left} 1fr"
        grid.styles.grid_rows = f"{self.margin_top} 1fr {self.margin_bottom}"

    def clear(self) -> None:
        """Clear the plot canvas."""
        self._datasets = []
        self._labels = []
        self._v_lines = []
        self._v_lines_labels = []
        self.refresh(layout=True)

    def plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        line_style: str = "white",
        hires_mode: HiResMode | None = None,
        label: str | None = None,
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
            label: A string with the label for the dataset. Defaults to None.
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
        self._labels.append(label)
        self.refresh(layout=True)

    def scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        marker: str = "o",
        marker_style: str = "white",
        hires_mode: HiResMode | None = None,
        label: str | None = None,
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
            label: A string with the label for the dataset. Defaults to None.
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
        self._labels.append(label)
        self.refresh(layout=True)

    def errorbar(
        self,
        x: ArrayLike,
        y: ArrayLike,
        xerr: ArrayLike | None = None,
        yerr: ArrayLike | None = None,
        marker: str = "",
        marker_style: str = "white",
        hires_mode: HiResMode | None = None,
        label: str | None = None,
    ) -> None:
        """Graph dataset using an error bar plot.

        If you supply hires_mode, the dataset will be plotted using one of the
        available high-resolution modes like 1x2, 2x2 or 2x8 pixel-per-cell
        characters.

        Args:
            x: An ArrayLike with the data values for the horizontal axis.
            y: An ArrayLike with the data values for the vertical axis.
            xerr: An ArrayLike with the error values for the horizontal axis,
                or None for no x errors. Defaults to None.
            yerr: An ArrayLike with the error values for the vertical axis,
                or None for no y errors. Defaults to None.
            marker: A string with the character to print as the marker.
            marker_style: A string with the style of the marker. Defaults to
                "white".
            hires_mode: A HiResMode enum or None to plot with the supplied
                marker. Defaults to None.
            label: A string with the label for the dataset. Defaults to None.
        """
        x, y = drop_nans_and_infs(np.array(x), np.array(y))

        # Convert error arrays to numpy arrays if provided
        xerr_array = np.array(xerr) if xerr is not None else np.zeros(shape=x.shape)
        yerr_array = np.array(yerr) if yerr is not None else np.zeros(shape=y.shape)

        self._datasets.append(
            ErrorBarPlot(
                x=x,
                y=y,
                xerr=xerr_array,
                yerr=yerr_array,
                marker=marker,
                marker_style=marker_style,
                hires_mode=hires_mode,
            )
        )
        self._labels.append(label)
        self.refresh(layout=True)

    def add_v_line(
        self, x: float, line_style: str = "white", label: str | None = None
    ) -> None:
        """Draw a vertical line on the plot.

        Args:
            x: The x-coordinate where the vertical line will be placed.
            line_style: A string with the style of the line. Defaults to "white".
            label: A string with the label for the line. Defaults to None.
        """
        self._v_lines.append(VLinePlot(x=x, line_style=line_style))
        self._v_lines_labels.append(label)
        self.refresh(layout=True)

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
        self.refresh(layout=True)

    def set_ylimits(self, ymin: float | None = None, ymax: float | None = None) -> None:
        """Set the limits of the y axis.

        Args:
            ymin: A float with the minimum y value or None for autoscaling.
                Defaults to None.
            ymax: A float with the maximum y value or None for autoscaling.
                Defaults to None.
        """
        self._user_y_min = ymin
        self._user_y_max = ymax
        self._auto_y_min = ymin is None
        self._auto_y_max = ymax is None
        self._y_min = ymin if ymin is not None else 0.0
        self._y_max = ymax if ymax is not None else 1.0
        self.refresh(layout=True)

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

    def set_xticks(self, ticks: Sequence[float] | None = None) -> None:
        """Set the x axis ticks.

        Use None for autoscaling, an empty list to hide the ticks.

        Args:
            ticks: An iterable with the tick values.
        """
        self._x_ticks = ticks

    def set_yticks(self, ticks: Sequence[float] | None = None) -> None:
        """Set the y axis ticks.

        Use None for autoscaling, an empty list to hide the ticks.

        Args:
            ticks: An iterable with the tick values.
        """
        self._y_ticks = ticks

    def set_x_formatter(self, formatter: AxisFormatter) -> None:
        """Set the formatter for the x axis.

        Args:
            formatter: An AxisFormatter instance to use for formatting x-axis ticks.
        """
        self._x_formatter = formatter

    def set_y_formatter(self, formatter: AxisFormatter) -> None:
        """Set the formatter for the y axis.

        Args:
            formatter: An AxisFormatter instance to use for formatting y-axis ticks.
        """
        self._y_formatter = formatter

    def show_legend(
        self,
        location: LegendLocation = LegendLocation.TOPRIGHT,
        is_visible: bool = True,
    ) -> None:
        """Show or hide the legend for the datasets in the plot.

        Args:
            is_visible: A boolean indicating whether to show the legend.
                Defaults to True.
        """
        self.query_one("#legend", Static).display = is_visible
        if not is_visible:
            return

        self._position_legend()

        legend_lines = []
        if isinstance(location, LegendLocation):
            self._legend_location = location
        else:
            raise TypeError(
                f"Expected LegendLocation, got {type(location).__name__} instead."
            )

        for label, dataset in zip(self._labels, self._datasets):
            if label is not None:
                if isinstance(dataset, ScatterPlot):
                    marker = (
                        dataset.marker
                        if dataset.hires_mode is None
                        else LEGEND_MARKER[dataset.hires_mode]
                    ).center(3)
                    style = dataset.marker_style
                elif isinstance(dataset, LinePlot):
                    marker = LEGEND_LINE[dataset.hires_mode]
                    style = dataset.line_style
                elif isinstance(dataset, ErrorBarPlot):
                    marker = (
                        dataset.marker
                        if dataset.hires_mode is None
                        else LEGEND_MARKER[dataset.hires_mode]
                    ).center(3)
                    style = dataset.marker_style
                else:
                    # unsupported dataset type
                    continue
                text = Text(marker)
                text.stylize(style)
                text.append(f" {label}")
                legend_lines.append(text.markup)

        for label, vline in zip(self._v_lines_labels, self._v_lines):
            if label is not None:
                marker = "│".center(3)
                style = vline.line_style
                text = Text(marker)
                text.stylize(style)
                text.append(f" {label}")
                legend_lines.append(text.markup)

        self.query_one("#legend", Static).update(
            Text.from_markup("\n".join(legend_lines))
        )

    def _position_legend(self) -> None:
        """Position the legend in the plot widget using absolute offsets.

        The position of the legend is calculated by checking the legend origin
        location (top left, bottom right, etc.) and an offset resulting from the
        user dragging the legend to another location. Then the nearest corner of
        the plot widget is determined and the legend is anchored to that corner
        and a new relative offset is determined. The end result is that the user
        can place the legend anywhere in the plot, but when the user resizes the
        plot the legend will stay fixed relative to the nearest corner.
        """

        position = (
            self._get_legend_origin_coordinates(self._legend_location)
            + self._legend_relative_offset
        )
        distances: dict[LegendLocation, float] = {
            location: self._get_legend_origin_coordinates(location).get_distance_to(
                position
            )
            for location in LegendLocation
        }
        nearest_location = min(distances, key=lambda loc: distances[loc])
        self._legend_location = nearest_location
        self._legend_relative_offset = position - self._get_legend_origin_coordinates(
            nearest_location
        )

        legend = self.query_one("#legend", Static)
        legend.offset = position

    def _get_legend_origin_coordinates(self, location: LegendLocation) -> Offset:
        """Calculate the (x, y) origin coordinates for positioning the legend.

        The coordinates are determined based on the legend's location (top-left,
        top-right, bottom-left, bottom-right), the size of the data rectangle,
        the length of the legend labels, and the margins and border spacing.
        User adjustments (dragging the legend to a different position) are _not_
        taken into account, but are applied later.

        Returns:
            A (x, y) tuple of ints representing the coordinates of the top-left
            corner of the legend within the plot widget.
        """
        canvas = self.query_one("#plot", Canvas)
        legend = self.query_one("#legend", Static)

        # Collect all labels that will appear in the legend
        all_labels = [label for label in self._labels if label is not None]
        all_labels.extend(
            [label for label in self._v_lines_labels if label is not None]
        )

        # markers and lines in the legend are 3 characters wide, plus a space, so 4
        max_length = 4 + max((len(s) for s in all_labels), default=0)

        if location in (LegendLocation.TOPLEFT, LegendLocation.BOTTOMLEFT):
            x0 = self.margin_left + 1
        else:
            # LegendLocation is TOPRIGHT or BOTTOMRIGHT
            x0 = self.margin_left + canvas.size.width - 1 - max_length
            # leave room for the border
            x0 -= legend.styles.border.spacing.left + legend.styles.border.spacing.right

        if location in (LegendLocation.TOPLEFT, LegendLocation.TOPRIGHT):
            y0 = self.margin_top + 1
        else:
            # LegendLocation is BOTTOMLEFT or BOTTOMRIGHT
            y0 = self.margin_top + canvas.size.height - 1 - len(all_labels)
            # leave room for the border
            y0 -= legend.styles.border.spacing.top + legend.styles.border.spacing.bottom
        return Offset(x0, y0)

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        """Refresh the widget.

        Args:
            regions: Specific regions to refresh.
            repaint: Whether to repaint the widget. Defaults to True.
            layout: Whether to refresh the layout. Defaults to False.
            recompose: Whether to recompose the widget. Defaults to False.

        Returns:
            The widget instance for method chaining.
        """
        if layout is True:
            self._needs_rerender = True
        return super().refresh(
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    def render(self) -> RenderResult:
        """Render the plot widget.

        Returns:
            An empty string as rendering is done on canvases.
        """
        if self._needs_rerender:
            self._needs_rerender = False
            self._render_plot()
        return ""

    def _render_plot(self) -> None:
        """Render all plot elements including datasets, axes, ticks, and labels."""
        try:
            if (canvas := self.query_one("#plot", Canvas))._canvas_size is None:
                return
        except NoMatches:
            # Refresh is called before the widget is composed
            return

        # clear canvas
        canvas.reset()

        # determine axis limits
        if self._datasets or self._v_lines:
            xs = [dataset.x for dataset in self._datasets]
            if self._v_lines:
                xs.append(np.array([vline.x for vline in self._v_lines]))
            ys = [dataset.y for dataset in self._datasets]
            if self._auto_x_min:
                non_empty_xs = [x for x in xs if len(x) > 0]
                if non_empty_xs:
                    self._x_min = float(np.min([np.min(x) for x in non_empty_xs]))
            if self._auto_x_max:
                non_empty_xs = [x for x in xs if len(x) > 0]
                if non_empty_xs:
                    self._x_max = float(np.max([np.max(x) for x in non_empty_xs]))
            if self._auto_y_min:
                non_empty_ys = [y for y in ys if len(y) > 0]
                if non_empty_ys:
                    self._y_min = float(np.min([np.min(y) for y in non_empty_ys]))
            if self._auto_y_max:
                non_empty_ys = [y for y in ys if len(y) > 0]
                if non_empty_ys:
                    self._y_max = float(np.max([np.max(y) for y in non_empty_ys]))

            if self._x_min == self._x_max:
                self._x_min -= 1e-6
                self._x_max += 1e-6
            if self._y_min == self._y_max:
                self._y_min -= 1e-6
                self._y_max += 1e-6

        # render datasets
        for dataset in self._datasets:
            if isinstance(dataset, LinePlot):
                self._render_line_plot(dataset)
            elif isinstance(dataset, ErrorBarPlot):
                self._render_errorbar_plot(dataset)
            elif isinstance(dataset, ScatterPlot):
                self._render_scatter_plot(dataset)

        # render axis, ticks and labels
        canvas.draw_rectangle_box(
            0,
            0,
            canvas.size.width - 1,
            canvas.size.height - 1,
            thickness=2,
            style=str(self.get_component_rich_style("plot--axis")),
        )
        # render vlines
        for vline in self._v_lines:
            self._render_v_line_plot(vline)
        # render tick marks and labels
        self._render_x_ticks()
        self._render_y_ticks()
        # render axis labels
        self._render_x_label()
        self._render_y_label()

    def _render_scatter_plot(self, dataset: ScatterPlot) -> None:
        """Render a scatter plot dataset on the canvas.

        Args:
            dataset: The scatter plot dataset to render.
        """
        canvas = self.query_one("#plot", Canvas)
        if dataset.hires_mode:
            hires_pixels = [
                self.get_hires_pixel_from_coordinate(xi, yi)
                for xi, yi in zip(dataset.x, dataset.y)
            ]
            canvas.set_hires_pixels(
                hires_pixels, style=dataset.marker_style, hires_mode=dataset.hires_mode
            )
        else:
            pixels = [
                self.get_pixel_from_coordinate(xi, yi)
                for xi, yi in zip(dataset.x, dataset.y)
            ]
            for pixel in pixels:
                canvas.set_pixel(
                    *pixel, char=dataset.marker, style=dataset.marker_style
                )

    def _render_errorbar_plot(self, dataset: ErrorBarPlot) -> None:
        """Render the error bars for an error bar plot.

        Both full-width and half-width characters are used for the errorbar.

        Args:
            dataset: The error bar plot dataset to render.
        """
        canvas = self.query_one("#plot", Canvas)

        # Render x error bars first
        if dataset.xerr is not None:
            # render error bars
            for xi, yi, xerr, yerr in zip(
                dataset.x, dataset.y, dataset.xerr, dataset.yerr
            ):
                center_px, center_py = self.get_pixel_from_coordinate(xi, yi)
                center_x, center_y = self.get_hires_pixel_from_coordinate(xi, yi)
                xe, ye = self.get_hires_pixel_from_coordinate(xi - xerr, yi - yerr)
                x_length = center_x - xe
                x_length_px = round(x_length)
                y_length = (
                    ye - center_y
                )  # canvas y-coordinates count from top to bottom
                y_length_px = round(y_length)

                # draw the full-width characters
                canvas.draw_line(
                    center_px - x_length_px,
                    center_py,
                    center_px + x_length_px,
                    center_py,
                    char="─",
                    style=dataset.marker_style,
                )
                canvas.draw_line(
                    center_px,
                    center_py - y_length_px,
                    center_px,
                    center_py + y_length_px,
                    char="│",
                    style=dataset.marker_style,
                )

                # render half-width characters if needed at the edges
                if (x_length - x_length_px) > 0.25:
                    canvas.set_pixel(
                        center_px - x_length_px - 1,
                        center_py,
                        char="╶",
                        style=dataset.marker_style,
                    )
                    canvas.set_pixel(
                        center_px + x_length_px + 1,
                        center_py,
                        char="╴",
                        style=dataset.marker_style,
                    )
                if (y_length - y_length_px) > 0.25:
                    canvas.set_pixel(
                        center_px,
                        center_py - y_length_px - 1,
                        char="╷",
                        style=dataset.marker_style,
                    )
                    canvas.set_pixel(
                        center_px,
                        center_py + y_length_px + 1,
                        char="╵",
                        style=dataset.marker_style,
                    )
                # if hires mode is off, render marker
                if dataset.hires_mode is None:
                    canvas.set_pixel(
                        center_px,
                        center_py,
                        dataset.marker or "┼",
                        style=dataset.marker_style,
                    )

            # render hires markers, if specified
            if dataset.hires_mode:
                self._render_scatter_plot(dataset)

    def _render_line_plot(self, dataset: LinePlot) -> None:
        """Render a line plot dataset on the canvas.

        Args:
            dataset: The line plot dataset to render.
        """
        canvas = self.query_one("#plot", Canvas)

        if dataset.hires_mode:
            hires_pixels = [
                self.get_hires_pixel_from_coordinate(xi, yi)
                for xi, yi in zip(dataset.x, dataset.y)
            ]
            coordinates = [
                (*hires_pixels[i - 1], *hires_pixels[i])
                for i in range(1, len(hires_pixels))
            ]
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

    def _render_v_line_plot(self, vline: VLinePlot) -> None:
        """Render a vertical line on the canvas.

        The vertical line is drawn from the top to the bottom of the scale
        rectangle and is connected to the scale rectangle.

        Args:
            vline: A VLinePlot dataclass instance containing the x-coordinate
                and line style.
        """
        canvas = self.query_one("#plot", Canvas)
        x, _ = self.get_pixel_from_coordinate(vline.x, 0)
        canvas.draw_line(
            x, 1, x, self._scale_rectangle.bottom - 1, style=vline.line_style, char="│"
        )
        style = str(self.get_component_rich_style("plot--axis"))
        canvas.set_pixel(x, 0, BOX_CHARACTERS[(0, 2, 2, 2)], style=style)
        canvas.set_pixel(
            x, self._scale_rectangle.bottom, BOX_CHARACTERS[(2, 2, 0, 2)], style=style
        )

    def _render_x_ticks(self) -> None:
        """Render tick marks and labels for the x-axis."""
        canvas = self.query_one("#plot", Canvas)
        bottom_margin = self.query_one("#margin-bottom", Canvas)
        bottom_margin.reset()

        x_ticks: Sequence[float]
        if self._x_ticks is None:
            x_ticks, x_labels = self._x_formatter.get_ticks_and_labels(
                self._x_min, self._x_max
            )
        else:
            x_ticks = self._x_ticks
            x_labels = self._x_formatter.get_labels_for_ticks(x_ticks)
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
                (self._scale_rectangle.bottom, (0, 0, 2, 0)),
            ]:
                new_pixel = self.combine_quad_with_pixel(quad, canvas, x, y)
                canvas.set_pixel(
                    x,
                    y,
                    new_pixel,
                    style=str(self.get_component_rich_style("plot--axis")),
                )
            bottom_margin.write_text(
                x + self.margin_left,
                0,
                f"[{self.get_component_rich_style('plot--tick')}]{label}",
                align,
            )

    def _render_y_ticks(self) -> None:
        """Render tick marks and labels for the y-axis."""
        canvas = self.query_one("#plot", Canvas)
        left_margin = self.query_one("#margin-left", Canvas)
        left_margin.reset()

        y_ticks: Sequence[float]
        if self._y_ticks is None:
            y_ticks, y_labels = self._y_formatter.get_ticks_and_labels(
                self._y_min, self._y_max
            )
        else:
            y_ticks = self._y_ticks
            y_labels = self._y_formatter.get_labels_for_ticks(y_ticks)
        # truncate y-labels to the left margin width
        y_labels = [label[: self.margin_left - 1] for label in y_labels]
        align = TextAlign.RIGHT
        for tick, label in zip(y_ticks, y_labels):
            if tick < self._y_min or tick > self._y_max:
                continue
            # only interested in the y-coordinate, set x to 0.0
            _, y = self.get_pixel_from_coordinate(0.0, tick)
            if tick == self._y_min:
                y += 1
            for x, quad in [
                # put ticks at left and right of scale rectangle
                (0, (0, 0, 0, 2)),
                (self._scale_rectangle.right, (0, 2, 0, 0)),
            ]:
                new_pixel = self.combine_quad_with_pixel(quad, canvas, x, y)
                canvas.set_pixel(
                    x,
                    y,
                    new_pixel,
                    style=str(self.get_component_rich_style("plot--axis")),
                )
            left_margin.write_text(
                self.margin_left - 2,
                y,
                f"[{self.get_component_rich_style('plot--tick')}]{label}",
                align,
            )

    def _render_x_label(self) -> None:
        """Render the x-axis label."""
        canvas = self.query_one("#plot", Canvas)
        margin = self.query_one("#margin-bottom", Canvas)
        margin.write_text(
            canvas.size.width // 2 + self.margin_left,
            2,
            f"[{self.get_component_rich_style('plot--label')}]{self._x_label}",
            TextAlign.CENTER,
        )

    def _render_y_label(self) -> None:
        """Render the y-axis label."""
        margin = self.query_one("#margin-top", Canvas)
        margin.write_text(
            self.margin_left - 2,
            0,
            f"[{self.get_component_rich_style('plot--label')}]{self._y_label}",
            TextAlign.CENTER,
        )

    def combine_quad_with_pixel(
        self, quad: tuple[int, int, int, int], canvas: Canvas, x: int, y: int
    ) -> str:
        """Combine a box-drawing quad with an existing pixel to create seamless connections.

        Args:
            quad: A tuple of 4 integers representing box drawing directions (top, right, bottom, left).
            canvas: The canvas containing the pixel.
            x: X-coordinate of the pixel.
            y: Y-coordinate of the pixel.

        Returns:
            A box-drawing character that combines both quads.
        """
        pixel = canvas.get_pixel(x, y)[0]
        for current_quad, v in BOX_CHARACTERS.items():
            if v == pixel:
                break
        else:
            raise ValueError(f"Pixel '{pixel}' is not a valid box drawing character.")
        new_quad = combine_quads(current_quad, quad)
        return BOX_CHARACTERS[new_quad]

    def get_pixel_from_coordinate(
        self, x: FloatScalar, y: FloatScalar
    ) -> tuple[int, int]:
        """Convert data coordinates to canvas pixel coordinates.

        Args:
            x: X-coordinate in data space.
            y: Y-coordinate in data space.

        Returns:
            A tuple of (x, y) pixel coordinates on the canvas.
        """
        return map_coordinate_to_pixel(
            x,
            y,
            self._x_min,
            self._x_max,
            self._y_min,
            self._y_max,
            region=self._scale_rectangle,
        )

    def get_hires_pixel_from_coordinate(
        self, x: FloatScalar, y: FloatScalar
    ) -> tuple[FloatScalar, FloatScalar]:
        """Convert data coordinates to high-resolution pixel coordinates.

        Args:
            x: X-coordinate in data space.
            y: Y-coordinate in data space.

        Returns:
            A tuple of (x, y) high-resolution pixel coordinates with sub-character precision.
        """
        return map_coordinate_to_hires_pixel(
            x,
            y,
            self._x_min,
            self._x_max,
            self._y_min,
            self._y_max,
            region=self._scale_rectangle,
        )

    def get_coordinate_from_pixel(self, x: int, y: int) -> tuple[float, float]:
        """Convert canvas pixel coordinates to data coordinates.

        Args:
            x: X-coordinate in pixel space.
            y: Y-coordinate in pixel space.

        Returns:
            A tuple of (x, y) coordinates in data space.
        """
        return map_pixel_to_coordinate(
            x,
            y,
            self._x_min,
            self._x_max,
            self._y_min,
            self._y_max,
            region=self._scale_rectangle,
        )

    def _zoom(self, event: MouseScrollDown | MouseScrollUp, factor: float) -> None:
        """Handle zoom operations centered on the mouse cursor position.

        Args:
            event: The mouse scroll event triggering the zoom.
            factor: The zoom factor to apply (positive for zoom in, negative for zoom out).
        """
        if not self._allow_pan_and_zoom:
            return

        if self.invert_mouse_wheel:
            factor *= -1

        if (offset := event.get_content_offset(self)) is not None:
            widget, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
            canvas = self.query_one("#plot", Canvas)
            if widget.id == "margin-bottom":
                offset = event.screen_offset - self.screen.get_offset(canvas)
            x, y = self.get_coordinate_from_pixel(offset.x, offset.y)
            if widget.id in ("plot", "margin-bottom"):
                self._auto_x_min = False
                self._auto_x_max = False
                self._x_min = (self._x_min + factor * x) / (1 + factor)
                self._x_max = (self._x_max + factor * x) / (1 + factor)
            if widget.id in ("plot", "margin-left"):
                self._auto_y_min = False
                self._auto_y_max = False
                self._y_min = (self._y_min + factor * y) / (1 + factor)
                self._y_max = (self._y_max + factor * y) / (1 + factor)
            self.post_message(
                self.ScaleChanged(
                    self, self._x_min, self._x_max, self._y_min, self._y_max
                )
            )
            self.refresh(layout=True)

    @on(MouseScrollDown)
    def zoom_in(self, event: MouseScrollDown) -> None:
        """Zoom into the plot when scrolling down.

        Args:
            event: The mouse scroll down event.
        """
        event.stop()
        self._zoom(event, ZOOM_FACTOR)

    @on(MouseScrollUp)
    def zoom_out(self, event: MouseScrollUp) -> None:
        """Zoom out of the plot when scrolling up.

        Args:
            event: The mouse scroll up event.
        """
        event.stop()
        self._zoom(event, -ZOOM_FACTOR)

    @on(MouseDown)
    def start_dragging_legend(self, event: MouseDown) -> None:
        """Start dragging the legend when clicked with left mouse button.

        Args:
            event: The mouse down event.
        """
        widget, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
        if event.button == 1 and widget.id == "legend":
            self._is_dragging_legend = True
            widget.add_class("dragged")
            event.stop()

    @on(MouseUp)
    def stop_dragging_legend(self, event: MouseUp) -> None:
        """Stop dragging the legend when left mouse button is released.

        Args:
            event: The mouse up event.
        """
        if event.button == 1 and self._is_dragging_legend:
            self._is_dragging_legend = False
            self.query_one("#legend").remove_class("dragged")
            event.stop()

    @on(MouseMove)
    def drag_with_mouse(self, event: MouseMove) -> None:
        """Handle mouse drag operations for panning the plot or moving the legend.

        Args:
            event: The mouse move event.
        """
        if not self._allow_pan_and_zoom:
            return
        if event.button == 0:
            # If no button is pressed, don't drag.
            return

        if self._is_dragging_legend:
            self._drag_legend(event)
        else:
            self._pan_plot(event)

    def _drag_legend(self, event: MouseMove) -> None:
        """Update legend position while dragging.

        Args:
            event: The mouse move event with drag delta information.
        """
        self._legend_relative_offset += event.delta
        self._position_legend()
        self.query_one("#legend").refresh(layout=True)

    def _pan_plot(self, event: MouseMove) -> None:
        """Pan the plot by adjusting axis limits based on mouse movement.

        Args:
            event: The mouse move event with drag delta information.
        """
        x1, y1 = self.get_coordinate_from_pixel(1, 1)
        x2, y2 = self.get_coordinate_from_pixel(2, 2)
        dx, dy = x2 - x1, y1 - y2

        assert event.widget is not None
        if event.widget.id in ("plot", "margin-bottom"):
            self._auto_x_min = False
            self._auto_x_max = False
            self._x_min -= dx * event.delta_x
            self._x_max -= dx * event.delta_x
        if event.widget.id in ("plot", "margin-left"):
            self._auto_y_min = False
            self._auto_y_max = False
            self._y_min += dy * event.delta_y
            self._y_max += dy * event.delta_y
        self.post_message(
            self.ScaleChanged(self, self._x_min, self._x_max, self._y_min, self._y_max)
        )
        self.refresh(layout=True)

    def action_reset_scales(self) -> None:
        """Reset the plot scales to the user-defined or auto-scaled limits."""
        self.set_xlimits(self._user_x_min, self._user_x_max)
        self.set_ylimits(self._user_y_min, self._user_y_max)
        self.post_message(
            self.ScaleChanged(self, self._x_min, self._x_max, self._y_min, self._y_max)
        )
        self.refresh()


def map_coordinate_to_pixel(
    x: FloatScalar,
    y: FloatScalar,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    region: Region,
) -> tuple[int, int]:
    """Map data coordinates to integer pixel coordinates within a region.

    Args:
        x: X-coordinate in data space.
        y: Y-coordinate in data space.
        xmin: Minimum x value in data space.
        xmax: Maximum x value in data space.
        ymin: Minimum y value in data space.
        ymax: Maximum y value in data space.
        region: The region defining the pixel space bounds.

    Returns:
        A tuple of (x, y) integer pixel coordinates.
    """
    x = floor(linear_mapper(x, xmin, xmax, region.x, region.right))
    # positive y direction is reversed
    y = ceil(linear_mapper(y, ymin, ymax, region.bottom - 1, region.y - 1))
    return x, y


def map_coordinate_to_hires_pixel(
    x: FloatScalar,
    y: FloatScalar,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    region: Region,
) -> tuple[FloatScalar, FloatScalar]:
    """Map data coordinates to floating-point high-resolution pixel coordinates.

    Args:
        x: X-coordinate in data space.
        y: Y-coordinate in data space.
        xmin: Minimum x value in data space.
        xmax: Maximum x value in data space.
        ymin: Minimum y value in data space.
        ymax: Maximum y value in data space.
        region: The region defining the pixel space bounds.

    Returns:
        A tuple of (x, y) floating-point pixel coordinates with sub-character precision.
    """
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
    """Map pixel coordinates to data coordinates within a region.

    This method takes the center of the pixel into account by adding 0.5
    to the pixel coordinates before mapping.

    Args:
        px: X-coordinate in pixel space.
        py: Y-coordinate in pixel space.
        xmin: Minimum x value in data space.
        xmax: Maximum x value in data space.
        ymin: Minimum y value in data space.
        ymax: Maximum y value in data space.
        region: The region defining the pixel space bounds.

    Returns:
        A tuple of (x, y) coordinates in data space.
    """
    x = linear_mapper(px + 0.5, region.x, region.right, xmin, xmax)
    # positive y direction is reversed
    y = linear_mapper(py + 0.5, region.bottom, region.y, ymin, ymax)
    return float(x), float(y)


def linear_mapper(
    x: FloatScalar | int,
    a: float | int,
    b: float | int,
    a_prime: float | int,
    b_prime: float | int,
) -> FloatScalar:
    """Perform linear mapping from range [a, b] to range [a_prime, b_prime].

    Args:
        x: The value to map from the source range.
        a: Start of the source range.
        b: End of the source range.
        a_prime: Start of the destination range.
        b_prime: End of the destination range.

    Returns:
        The mapped value in the destination range.
    """
    return a_prime + (x - a) * (b_prime - a_prime) / (b - a)


def drop_nans_and_infs(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Drop NaNs and Infs from x and y arrays.

    Args:
        x: An array with the data values for the horizontal axis.
        y: An array with the data values for the vertical axis.

    Returns:
        A tuple of arrays with NaNs and Infs removed.
    """
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    return x[mask], y[mask]
