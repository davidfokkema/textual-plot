from datetime import datetime, timedelta, timezone

import numpy as np
from textual.app import App, ComposeResult

from textual_plot import PlotWidget
from textual_plot.axis_formatter import DateTimeFormatter


class DateTimeApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        t0 = datetime(2020, 1, 1)
        t1 = datetime(2030, 12, 31)
        x = np.linspace(t0.timestamp(), t1.timestamp(), 1000)
        y = np.random.normal(1.0, 1.0, size=1000)
        plot.scatter(x, y)
        plot.set_x_formatter(DateTimeFormatter())


DateTimeApp().run()
