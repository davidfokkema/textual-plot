from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.geometry import Size
from textual.strip import Strip
from textual.widget import Widget


class Canvas(Widget):
    _canvas_size: Size | None = None

    def _on_resize(self) -> None:
        if self._canvas_size is None:
            self._canvas_size = self.size

    def render_lines(self, crop):
        if self._canvas_size is None:
            return []
        return super().render_lines(crop)

    def render_line(self, y) -> Strip:
        return Strip(
            [Segment(self._canvas_size.width * "/", style=Style.parse("blue on black"))]
        )


class DemoApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Canvas()


if __name__ == "__main__":
    DemoApp().run()
