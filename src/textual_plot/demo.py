from __future__ import annotations

import importlib.resources
import itertools

import numpy as np
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid
from textual.widgets import Footer, Header, Label, TabbedContent, TabPane
from textual_hires_canvas import HiResMode

from textual_plot import PlotWidget


class SpectrumPlot(Container):
    BINDINGS = [("m", "cycle_modes", "Cycle Modes")]

    _modes = itertools.cycle(
        [HiResMode.QUADRANT, HiResMode.BRAILLE, None, HiResMode.HALFBLOCK]
    )
    mode = next(_modes)

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        # Read CSV data included with this package
        self.spectrum_csv = importlib.resources.read_text(
            "textual_plot.resources", "morning-spectrum.csv"
        ).splitlines()

        # plot the spectrum and set ymin limit once
        self.plot_spectrum()
        self.query_one(PlotWidget).set_ylimits(ymin=0)

    def plot_spectrum(self) -> None:
        x, y = np.genfromtxt(
            self.spectrum_csv,
            delimiter=",",
            names=True,
            unpack=True,
        )

        plot = self.query_one(PlotWidget)
        plot.clear()
        plot.plot(x, y, hires_mode=self.mode)
        plot.add_v_line(589, "blue", "Sodium line")
        plot.set_xlabel("Wavelength (nm)")
        plot.set_ylabel("Intensity")
        plot.show_legend()

    def action_cycle_modes(self) -> None:
        self.mode = next(self._modes)
        self.plot_spectrum()


class SinePlot(Container):
    BINDINGS = [("c", "clear", "Clear")]

    N: int = 1

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 24, self.plot_moving_sines, pause=True)

    def on_show(self) -> None:
        self._timer.resume()

    def on_hide(self) -> None:
        self._timer.pause()

    def action_clear(self) -> None:
        self.N = 1

    def plot_moving_sines(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        x = np.arange(0, self.N * 0.1, 0.1)
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + 3),
            line_style="yellow",
            hires_mode=HiResMode.BRAILLE,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + 2),
            line_style="green",
            hires_mode=HiResMode.QUADRANT,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + 1),
            line_style="red3",
            hires_mode=HiResMode.HALFBLOCK,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x),
            line_style="blue",
            hires_mode=None,
        )

        plot.set_ylimits(0, 20)
        self.N += 1


class MultiPlot(Grid):
    BINDINGS = [Binding("r", "reset_scales", "Reset scales", priority=True)]

    DEFAULT_CSS = """
        MultiPlot {
            grid-size: 2 4;
            grid-rows: auto 1fr;

            Label {
                text-align: center;
                text-style: bold;
                padding: 1 2 0 2;
                width: 100%;
            }

            PlotWidget {
      
            }
        }
    """

    def compose(self) -> ComposeResult:
        yield Label("f(x) = x")
        yield Label("f(x) = x ** 2")
        yield PlotWidget(id="x")
        yield PlotWidget(id="x-squared")
        yield Label("f(x) = 1 / |1 + x|")
        yield Label("f(x) = sqrt(x)")
        yield PlotWidget(id="one-over-x")
        yield PlotWidget(id="sqrt-x")

    def on_mount(self) -> None:
        for plot in self.query(PlotWidget):
            plot.margin_left = 8
            plot.margin_top = 0
            plot.margin_bottom = 1
        self.plot()

    def plot(self) -> None:
        plot = self.query_one("#x", PlotWidget)
        x = np.linspace(plot._x_min, plot._x_max, 101)
        plot.clear()
        plot.plot(x, x, hires_mode=HiResMode.BRAILLE)

        plot = self.query_one("#x-squared", PlotWidget)
        plot.clear()
        plot.plot(x, x**2, hires_mode=HiResMode.BRAILLE)

        plot = self.query_one("#one-over-x", PlotWidget)
        plot.clear()
        plot.plot(x, 1 / abs(1 + x), hires_mode=HiResMode.BRAILLE)

        plot = self.query_one("#sqrt-x", PlotWidget)
        plot.clear()
        plot.plot(x, np.sqrt(x), hires_mode=HiResMode.BRAILLE)

    @on(PlotWidget.ScaleChanged)
    def adjust_scales(self, event: PlotWidget.ScaleChanged) -> None:
        for plot in self.query(PlotWidget):
            plot.set_xlimits(event.x_min, event.x_max)
            plot.set_ylimits()
        self.plot()

    def action_reset_scales(self) -> None:
        for plot in self.query(PlotWidget):
            plot.set_xlimits(0.0, 1.0)
            plot.set_ylimits()
        self.plot()


class DemoApp(App[None]):
    AUTO_FOCUS = "SinePlot > PlotWidget"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with TabbedContent():
            with TabPane("Moving sines", id="sines"):
                yield SinePlot()
            with TabPane("Daytime spectrum", id="spectrum"):
                yield SpectrumPlot()
            with TabPane("Multiplot", id="multiplot"):
                yield MultiPlot()


def main() -> None:
    app = DemoApp()
    app.run()


if __name__ == "__main__":
    main()
