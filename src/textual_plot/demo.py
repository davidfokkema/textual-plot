from __future__ import annotations

import importlib.resources
import itertools

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, TabbedContent, TabPane
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
        plot.set_xlabel("Wavelength (nm)")
        plot.set_ylabel("Intensity")

    def action_cycle_modes(self) -> None:
        self.mode = next(self._modes)
        self.plot_spectrum()


class SinePlot(Container):
    _phi: float = 0.0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 24, self.plot_moving_sines, pause=True)

    def on_show(self) -> None:
        self._timer.resume()

    def on_hide(self) -> None:
        self._timer.pause()

    def plot_moving_sines(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        x = np.linspace(0, 10, 41)
        y = x**2 / 3.5
        plot.scatter(
            x,
            y,
            marker_style="blue",
            # marker="*",
            hires_mode=HiResMode.QUADRANT,
        )
        x = np.linspace(0, 10, 200)
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi),
            line_style="blue",
            hires_mode=None,
        )

        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi + 1),
            line_style="red3",
            hires_mode=HiResMode.HALFBLOCK,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi + 2),
            line_style="green",
            hires_mode=HiResMode.QUADRANT,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi + 3),
            line_style="yellow",
            hires_mode=HiResMode.BRAILLE,
        )

        self._phi += 0.1


class DemoApp(App[None]):
    AUTO_FOCUS = "PlotWidget"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with TabbedContent():
            with TabPane("Daytime spectrum"):
                yield SpectrumPlot()
            with TabPane("Moving sines"):
                yield SinePlot()


def main() -> None:
    app = DemoApp()
    app.run()


if __name__ == "__main__":
    main()
