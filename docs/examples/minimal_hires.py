from textual.app import App, ComposeResult

from textual_plot import HiResMode, PlotWidget


class MinimalApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.plot(
            x=[0, 1, 2, 3, 4],
            y=[0, 1, 4, 9, 16],
            hires_mode=HiResMode.BRAILLE,
            line_style="bright_yellow on blue3",
        )


MinimalApp().run()
