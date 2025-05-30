# A native plotting widget for Textual apps

[Textual](https://www.textualize.io/) is an excellent Python framework for building applications in the terminal, or on the web. This library provides a plot widget which your app can use to plot all kinds of quantitative data. So, no pie charts, sorry. The widget support scatter plots and line plots, and can also draw using _high-resolution_ characters like unicode half blocks, quadrants and 8-dot Braille characters. It may still be apparent that these are drawn using characters that take up a full block in the terminal, especially when plot series overlap. However, the use of these characters can reduce the line thickness and improve the resolution tremendously.

## Screenshots

![screenshot of day-time spectrum](docs/images/screenshot-spectrum.png)

![screenshot of moving sines](docs/images/screenshot-moving-sines.png)

![video of plot demo](https://github.com/user-attachments/assets/dd725fdc-e182-4bed-8951-5899bdb99a20)

The _daytime spectrum_ dataset shows the visible-light spectrum recorded by an Ocean Optics USB2000+ spectrometer using the [DeadSea Optics](https://github.com/davidfokkema/deadsea-optics) software. It was taken in the morning while the detector was facing my office window.

## Features

- Line plots
- Scatter plots
- Automatic scaling and tick placement at _nice_ intervals (1, 2, 5, etc.)
- Axes labels
- High-resolution modes using unicode half blocks (1x2), quadrants (2x2) and braille (2x8) characters
- Mouse support for _zooming_ (mouse scrolling) and _panning_ (mouse dragging)
- Horizontal- or vertical-only zooming and panning when the mouse cursor is in the plot margins

## Running the demo / installation

Using [uv](https://astral.sh/uv/):
```console
uvx textual-plot
```

Using [pipx](https://pipx.pypa.io/):
```console
pipx run textual-plot
```

Install the package with either
```console
uv tool install textual-plot
```
or
```console
pipx install textual-plot
```
Alternatively, install the package with `pip` (please, use virtual environments) and run the demo:
```console
pip install textual-plot
```

In all cases, you can run the demo with
```console
textual-plot
```

## Tutorial

A minimal example is shown below:
![screenshot of minimal example](docs/images/screenshot-minimal.png)
```python
from textual.app import App, ComposeResult

from textual_plot import PlotWidget


class MinimalApp(App[None]):
    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.plot(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])


MinimalApp().run()
```
You include a `PlotWidget` in your compose method and after your UI has finished composing, you can start plotting data. The `plot()` method takes `x` and `y` data which should be array-like. It can be lists, or NumPy arrays, or really anything that can be turned into a NumPy array which is what's used internally. The `plot()` method further accepts a `line_style` argument which accepts Textual styles like `"white"`, `"red on blue3"`, etc. For standard low-resolution plots, it does not make much sense to specify a background color since the text character used for plotting is a full block filling an entire cell.

### High-resolution plotting

The plot widget supports high-resolution plotting where the character does not take up the full cell:

![screenshot of minimal hires example](docs/images/screenshot-minimal-hires.png)

```python
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
```
Admittedly, you'll be mostly plotting with foreground colors only. The plot widget supports four high-resolution modes: `Hires.BRAILLE` (2x8), `HiRes.HALFBLOCK` (1x2) and `HiRes.QUADRANT` (2x2) where the size between brackets is the number of 'pixels' inside a single cell.

### Scatter plots

To create scatter plots, use the `scatter()` method, which accepts a `marker` argument which can be any unicode character (as long as it is one cell wide, which excludes many emoji characters and non-Western scripts):
![screenshot of scatter plot](docs/images/screenshot-scatter.png)
```python
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
```

### The full demo code

Finally, the code of the demo is given below, showing how you can handle multiple plots and updating 'live' data:
```python
import importlib.resources
import itertools

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, TabbedContent, TabPane
from textual_hires_canvas import HiResMode

from textual_plot import PlotWidget


class SpectrumPlot(Container):
    BINDINGS = [("m", "cycle_modes", "Cycle Modes")]

    _modes = itertools.cycle(
        [HiResMode.QUADRANT, HiResMode.BRAILLE, None, HiResMode.HALFBLOCK]
    )
    mode = next(_modes)

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        # Read CSV data included with this package
        self.spectrum_csv = importlib.resources.read_text(
            "textual_plot.resources", "morning-spectrum.csv"
        ).splitlines()

        # plot the spectrum and set ymin limit once
        self.plot_spectrum()
        self.query_one(PlotWidget).set_ylimits(ymin=0)

    def plot_spectrum(self) -> None:
        x, y = np.genfromtxt(
            self.spectrum_csv,
            delimiter=",",
            names=True,
            unpack=True,
        )

        plot = self.query_one(PlotWidget)
        plot.clear()
        plot.plot(x, y, hires_mode=self.mode)
        plot.set_xlabel("Wavelength (nm)")
        plot.set_ylabel("Intensity")

    def action_cycle_modes(self) -> None:
        self.mode = next(self._modes)
        self.plot_spectrum()


class SinePlot(Container):
    _phi: float = 0.0

    def compose(self) -> ComposeResult:
        yield PlotWidget()

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 24, self.plot_moving_sines, pause=True)

    def on_show(self) -> None:
        self._timer.resume()

    def on_hide(self) -> None:
        self._timer.pause()

    def plot_moving_sines(self) -> None:
        plot = self.query_one(PlotWidget)
        plot.clear()
        x = np.linspace(0, 10, 41)
        y = x**2 / 3.5
        plot.scatter(
            x,
            y,
            marker_style="blue",
            # marker="*",
            hires_mode=HiResMode.QUADRANT,
        )
        x = np.linspace(0, 10, 200)
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi),
            line_style="blue",
            hires_mode=None,
        )

        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi + 1),
            line_style="red3",
            hires_mode=HiResMode.HALFBLOCK,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi + 2),
            line_style="green",
            hires_mode=HiResMode.QUADRANT,
        )
        plot.plot(
            x=x,
            y=10 + 10 * np.sin(x + self._phi + 3),
            line_style="yellow",
            hires_mode=HiResMode.BRAILLE,
        )

        self._phi += 0.1


class DemoApp(App[None]):
    AUTO_FOCUS = "PlotWidget"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with TabbedContent():
            with TabPane("Daytime spectrum"):
                yield SpectrumPlot()
            with TabPane("Moving sines"):
                yield SinePlot()


def main():
    app = DemoApp()
    app.run()


if __name__ == "__main__":
    main()
```

## List of important plot widget methods

- `clear()`: clear the plot.
- `plot(x, y, line_style, hires_mode, label)`: plot a dataset with a line using the specified linestyle and high-resolution mode.
- `scatter(x, y, marker, marker_style, hires_mode, label)`: plot a dataset with markers using the specified marker, marker style and high-resolution mode.
- `set_xlimits(xmin, xmax)`: set the x-axis limits. `None` means autoscale.
- `set_ylimits(xmin, xmax)`: set the y-axis limits. `None` means autoscale.
- `set_xticks(ticks)`: manually specify x-axis tick locations.
- `set_yticks(ticks)`: manually specify y-axis tick locations.
- `set_xlabel(label)`: set the x-axis label.
- `set_ylabel(label)`: set the y-axis label.
- `show_legend(location, is_visible)`: show or hide the plot legend.

Various other methods exist, mostly for coordinate transformations and handling UI events to zoom and pan the plot.

## Alternatives

[Textual-plotext](https://github.com/Textualize/textual-plotext) uses the [plotext](https://github.com/piccolomo/plotext) library which has more features than this library. However, it does not support interactive zooming or panning and the tick placement isn't as nice since it simply divides up the axes range into a fixed number of intervals giving values like 0, 123.4, 246.8, etc.

## Roadmap

The performance can be much improved, but we're working on that. Next, we'll work on adding some features like date axes. This will (probably) not turn into a general do-it-all plotting library. We focus first on handling quantitative data in the context of physics experiments. If you'd like to see features added, do let us know. And if a PR is of good quality and is a good fit for the API, we'd love to handle more use cases beyond physics. And who knows, maybe this _will_ turn into a general plotting library!