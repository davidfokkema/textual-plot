import random
import time

from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.strip import Strip
from textual.widget import Widget


class PlotWidget(Widget):
    def on_mount(self) -> None:
        self._count = 0
        self._t0 = time.monotonic()
        self.set_interval(1, self.show_refresh_rate)

    def render_lines(self, crop):
        self._count += 1
        return super().render_lines(crop)

    def show_refresh_rate(self) -> None:
        rate = self._count / (time.monotonic() - self._t0)
        self._count = 0
        self._t0 = time.monotonic()
        self.notify(f"Refresh rate: {rate:.1f}")

    def render_line(self, y: int) -> Strip:
        """Render a line of the widget. y is relative to the top of the widget."""
        segments = [
            Segment(random.choice([".", ",", "o", "x"])) for x in range(self.size.width)
        ]
        # segments = [
        #     Segment(
        #         "".join(
        #             [
        #                 random.choice([".", ",", "o", "x"])
        #                 for x in range(self.size.width)
        #             ]
        #         )
        #     )
        # ]
        strip = Strip(segments, cell_length=self.size.width)

        return strip


class DemoApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self.set_interval(1 / 600, self.plot_refresh)

    def plot_refresh(self) -> None:
        self.query_one(PlotWidget).refresh()


if __name__ == "__main__":
    app = DemoApp()
    app.run()
