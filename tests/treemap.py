"""Demo of treemap plot in textual-plot."""

from textual.app import App, ComposeResult

from textual_plot import PlotWidget


class TreemapApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        values = [500, 433, 78, 25, 25, 7]
        labels = ["A", "B", "C", "D", "E", "F"]
        styles = ["red", "blue", "green", "yellow", "cyan", "magenta"]
        plot.treemap(
            values,
            labels=labels,
            styles=styles,
            padding=1,
            label="Categories",
        )
        plot.show_legend()


TreemapApp().run()
