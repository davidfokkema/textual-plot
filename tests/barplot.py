from textual.app import App, ComposeResult

from textual_plot import HiResMode, PlotWidget


class BarPlotApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        x = [1, 2, 3, 4, 5]
        y = [10.2, 8.3, 7.5, 9.1, 9]
        styles = ["red", "blue", "green", "white", "yellow"]
        plot.bar(
            x,
            y,
            bar_style=styles,
            width=0.8,
            label="Fancy bars",
            hires_mode=HiResMode.BRAILLE,
        )
        plot.show_legend()


BarPlotApp().run()
