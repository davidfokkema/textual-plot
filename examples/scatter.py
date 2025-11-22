import numpy as np
from textual.app import App, ComposeResult

from textual_plot import PlotWidget


class MinimalApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        rng = np.random.default_rng(seed=4)
        plot = self.query_one(PlotWidget)

        x = np.linspace(0, 10, 21)
        y = 0.2 * x - 1 + rng.normal(loc=0.0, scale=0.2, size=len(x))
        plot.scatter(x, y, marker="⦿")


MinimalApp().run()
