# A native plotting widget for Textual apps

[Textual](https://www.textualize.io/) is an excellent Python framework for building applications in the terminal, or on the web. This library provides a plot widget which your app can use to plot all kinds of quantitative data. So, no pie charts, sorry. The widget support scatter plots and line plots, and can also draw using _high-resolution_ characters like unicode half blocks, quadrants and 8-dot Braille characters. It may still be apparent that these are drawn using characters that take up a full block in the terminal, especially when plot series overlap. However, the use of these characters can reduce the line thickness and improve the resolution tremendously.

## Screenshots

![screenshot of day-time spectrum](https://raw.githubusercontent.com/davidfokkema/textual-plot/refs/heads/main/docs/images/screenshot-spectrum.png)

![screenshot of moving sines](https://raw.githubusercontent.com/davidfokkema/textual-plot/refs/heads/main/docs/images/screenshot-moving-sines.png)

![screenshot of error bars](https://raw.githubusercontent.com/davidfokkema/textual-plot/refs/heads/main/docs/images/screenshot-errorbars.png)

![screenshot of bar plot](https://raw.githubusercontent.com/davidfokkema/textual-plot/refs/heads/main/docs/images/screenshot-barplot.png)

![video of plot demo](https://github.com/user-attachments/assets/dd725fdc-e182-4bed-8951-5899bdb99a20)

The _daytime spectrum_ dataset shows the visible-light spectrum recorded by an Ocean Optics USB2000+ spectrometer using the [DeadSea Optics](https://github.com/davidfokkema/deadsea-optics) software. It was taken in the morning while the detector was facing my office window.

## Features

- Line plots
- Scatter plots
- Scatter plots with errorbars
- Bar plots, also for categorical data
- Vertical lines
- Automatic scaling and tick placement at _nice_ intervals (1, 2, 5, etc.)
- Axes labels
- Axis formatters for numbers, durations and categories
- Support for custom axis formatters
- High-resolution modes using unicode half blocks (1x2), quadrants (2x2) and braille (2x8) characters
- Mouse support for _zooming_ (mouse scrolling) and _panning_ (mouse dragging)
- Horizontal- or vertical-only zooming and panning when the mouse cursor is in the plot margins
- Keyboard bindings for zooming, panning, and resetting the scale.

## Running the demo / installation

Using [uv](https://astral.sh/uv/):
```console
uvx textual-plot
```

Using [pipx](https://pipx.pypa.io/):
```console
pipx run textual-plot
```

Install the package with `pip` (please, use virtual environments):
```console
pip install textual-plot
```

## Tutorial

A minimal example is shown below:
![screenshot of minimal example](https://raw.githubusercontent.com/davidfokkema/textual-plot/refs/heads/main/docs/images/screenshot-minimal.png)
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

A longer tutorial can be found at [here](https://davidfokkema.github.io/textual-plot/#tutorial).

## Reference

A reference of all available methods can be found [here](https://davidfokkema.github.io/textual-plot/reference/).

## Alternatives

[Textual-plotext](https://github.com/Textualize/textual-plotext) uses the [plotext](https://github.com/piccolomo/plotext) library which has more features than this library. However, it does not support interactive zooming or panning and the tick placement isn't as nice since it simply divides up the axes range into a fixed number of intervals giving values like 0, 123.4, 246.8, etc. The plotext library hasn't been updated since September, 2024 and hadn't seen significant updates since December, 2023. The author is working on a new version, a full rewrite, since at least [October 2023](https://github.com/piccolomo/plotext/issues/184#issuecomment-1751548546). If you're looking for a library to use for your [Textual](https://textual.textualize.io) application, this is it.

## Roadmap

We'll work on adding some features like datetime axes, and some quality-of-life improvements like [even tick spacing](https://github.com/davidfokkema/textual-plot/issues/14). This will (probably) not turn into a general do-it-all plotting library. We focus first on handling quantitative data in the context of physics experiments. If you'd like to see features added, do let us know. And if a PR is of good quality and is a good fit for the API, we'd love to handle more use cases beyond physics. And who knows, maybe this _will_ turn into a general plotting library!