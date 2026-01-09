from textual.app import App, ComposeResult

from textual_plot import PlotWidget


class BarPlotApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 7, 9, 4]
        styles = ["red", "blue", "green", "white", "yellow"]
        plot.bar(x, y, bar_style=styles, width=0.5)


BarPlotApp().run()
