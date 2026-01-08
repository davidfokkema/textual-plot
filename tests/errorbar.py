import numpy as np
from textual.app import App, ComposeResult

from textual_plot import HiResMode, PlotWidget


class ErrorBarApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        x = np.linspace(1.0, 9.0, 5)
        y = x / 2
        xerr = x / 3
        yerr = y / 3
        plot.errorbar(x, y, xerr, yerr)
        plot.set_xlimits(0, 10)
        plot.set_ylimits(0, 5)


ErrorBarApp().run()
