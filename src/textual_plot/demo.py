import numpy as np
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header
from textual_hires_canvas import HiResMode

from textual_plot import PlotWidget


class DemoApp(App[None]):
    _phi: float = 0.0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield PlotWidget()

    def on_mount(self) -> None:
        # self.set_interval(1 / 24, self.plot_refresh)
        # self.plot_refresh()
        plot = self.query_one(PlotWidget)
        x, y = np.genfromtxt(
            "morning-spectrum.csv", delimiter=",", names=True, unpack=True
        )
        plot.plot(x, y, hires_mode=HiResMode.QUADRANT)
        plot.set_ylimits(ymin=0)
        plot.set_xlabel("Wavelength (nm)")
        plot.set_ylabel("Intensity")

    def plot_refresh(self) -> None:
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


if __name__ == "__main__":
    app = DemoApp()
    app.run()
