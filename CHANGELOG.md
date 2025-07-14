# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-07-14

### Added

- Add a plot legend, which you can target using CSS (#9).
- Added an `invert_mouse_wheel` parameter for switching zoom direction from [@ddkasa](https://github.com/ddkasa) (#13).

### Changed

- Converted `allow_pan_and_zoom` to a keyword only argument (#13).
- The `_scale_rectangle` attribute is now a part of PlotWidget and no longer of Canvas.

### Fixed

- Lots and lots of type annotations.

## [0.6.1] - 2025-05-24

### Fixed

- Truncate the y-axis tick label to avoid the leftmost (most significant) digit being clipped (#4).

## [0.6.0] - 2025-05-24

### Changed

- Lower Python version requirements back to 3.10.

### Fixed

- Zooming the plot no longer scrolls the container the plot is in.

## [0.5.0] - 2025-05-01

### Fixed

- Fix crash on newer textual / textual-hires-canvas when render() runs before all variables are initialised.

## [0.4.0] - 2025-03-16

### Fixed

- Fix inter-tick spacing becoming an order of magnitude too large due to negative indexing.
- Improve zooming performance by delaying refresh (batching render calls) and
  needs_rerender flag, performing the render only once in a batch.
- Fix invisible plot on first focus event.

## [0.3.0] - 2025-03-11

### Added

- `PlotWidget` now has name, classes and disabled parameters.
- Added `allow_pan_and_zoom` parameter to allow or disable panning and zooming the plot.
- Added setting x and y ticks to specific values, or an empty list to hide the ticks.

## [0.2.0] - 2025-03-01

### Added

- Post a ScaleChanged message when the user zooms, pans, or resets the scale.

### Fixed

- Fix crash when data contained NaN of Inf values.
- Fix crash when y_min and y_max are identical, i.e. when plotting a constant value.

## [0.1.1] - 2025-02-14

### Fixed

- Fix missing csv file for demo.

## [0.1.0] - 2025-02-14

Initial release. 📈🎉
