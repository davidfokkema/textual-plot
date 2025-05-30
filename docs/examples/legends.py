import numpy as np
from textual.app import App, ComposeResult

from textual_plot import HiResMode, LegendLocation, PlotWidget


class LegendsApp(App[None]):
    BINDINGS = [("t", "toggle_legend", "Toggle legend")]

    show_legend: bool = True

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        rng = np.random.default_rng(seed=4)
        plot = self.query_one(PlotWidget)

        x = np.linspace(0, 10, 71)
        y = 0.5 * x - 1 + rng.normal(loc=0.0, scale=0.2, size=len(x))
        plot.scatter(x, y, marker="⦿", label="Series 1")
        plot.scatter(
            x, y + 0.5, marker="⦿", label="Series 1.5", hires_mode=HiResMode.QUADRANT
        )
        plot.scatter(x, y + 1, label="Series 2", marker_style="bold italic green")
        plot.plot(x, y + 2, label="Series 3", line_style="red")
        plot.plot(
            x,
            y + 3,
            label="Series 3",
            line_style="blue",
            hires_mode=HiResMode.BRAILLE,
        )
        plot.plot(
            x,
            y + 4,
            # label="Series 4",
            hires_mode=HiResMode.HALFBLOCK,
        )
        plot.plot(
            x,
            y + 5,
            label="Series 5",
            line_style="bold italic cyan",
            hires_mode=HiResMode.QUADRANT,
        )
        plot.show_legend(location=LegendLocation.TOPLEFT)

    def action_toggle_legend(self) -> None:
        plot = self.query_one(PlotWidget)
        self.show_legend = not self.show_legend
        plot.show_legend(is_visible=self.show_legend)


LegendsApp().run()
