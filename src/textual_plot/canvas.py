from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.strip import Strip
from textual.widget import Widget


class Canvas(Widget):
    def render_line(self, y) -> Strip:
        return Strip(
            [Segment(self.size.width * "/", style=Style.parse("blue on black"))]
        )


class DemoApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Canvas()


if __name__ == "__main__":
    DemoApp().run()
