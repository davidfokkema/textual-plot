"""Axis formatters for PlotWidget.

This module provides classes for formatting axis ticks and labels in plots.
Different formatters can be used for different types of data (numeric, datetime, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil, floor, log10
from typing import Sequence

import numpy as np


class AxisFormatter(ABC):
    """Abstract base class for axis formatters.

    Axis formatters are responsible for generating tick positions and labels
    for plot axes. Subclasses can implement different formatting strategies
    for different types of data (e.g., numeric, datetime, logarithmic).
    """

    @abstractmethod
    def get_ticks(self, min_: float, max_: float, max_ticks: int = 8) -> list[float]:
        """Generate tick positions.

        Args:
            min_: Minimum value of the axis range.
            max_: Maximum value of the axis range.
            max_ticks: Maximum number of ticks to generate. Defaults to 8.

        Returns:
            A list of tick positions (as floats).
        """
        pass

    @abstractmethod
    def get_labels_for_ticks(self, ticks: Sequence[float]) -> list[str]:
        """Generate formatted labels for given tick positions.

        Args:
            ticks: A sequence of tick positions to be formatted.

        Returns:
            A list of formatted tick labels as strings.
        """
        pass

    def get_ticks_and_labels(
        self, min_: float, max_: float, max_ticks: int = 8
    ) -> tuple[list[float], list[str]]:
        """Generate tick positions and their corresponding labels.

        This is a convenience method that calls get_ticks() followed by
        get_labels_for_ticks().

        Args:
            min_: Minimum value of the axis range.
            max_: Maximum value of the axis range.
            max_ticks: Maximum number of ticks to generate. Defaults to 8.

        Returns:
            A tuple containing:
                - A list of tick positions (as floats)
                - A list of formatted tick labels (as strings)
        """
        ticks = self.get_ticks(min_, max_, max_ticks)
        labels = self.get_labels_for_ticks(ticks)
        return ticks, labels


class NumericAxisFormatter(AxisFormatter):
    """Formatter for numeric axes with nice intervals (1, 2, 5, 10, etc.).

    This formatter generates ticks at "nice" intervals like 1, 2, 5, 10, 20, 50,
    100, etc., which are visually pleasing and easy to read.
    """

    def get_ticks(self, min_: float, max_: float, max_ticks: int = 8) -> list[float]:
        """Generate tick values at nice intervals (1, 2, 5, 10, etc.).

        Args:
            min_: Minimum value of the range.
            max_: Maximum value of the range.
            max_ticks: Maximum number of ticks to generate. Defaults to 8.

        Returns:
            A list of tick values.
        """
        delta_x = max_ - min_
        tick_spacing = delta_x / 5
        power = floor(log10(tick_spacing))
        approx_interval = tick_spacing / 10**power
        intervals = np.array([1.0, 2.0, 5.0, 10.0])

        idx = intervals.searchsorted(approx_interval)
        interval = (intervals[idx - 1] if idx > 0 else intervals[0]) * 10**power
        if delta_x // interval > max_ticks:
            interval = intervals[idx] * 10**power
        ticks = [
            float(t)
            for t in np.arange(
                ceil(min_ / interval) * interval, max_ + interval / 2, interval
            )
        ]
        return ticks

    def get_labels_for_ticks(self, ticks: Sequence[float]) -> list[str]:
        """Generate formatted labels for given tick values.

        The number of decimal places is automatically determined from the tick spacing.

        Args:
            ticks: A list of tick values to be formatted.

        Returns:
            A list of formatted tick labels as strings.
        """
        if not ticks:
            return []
        # Automatically determine decimals from tick spacing
        if len(ticks) >= 2:
            power = floor(log10(ticks[1] - ticks[0]))
        else:
            power = 0
        decimals = -min(0, power)
        tick_labels = [f"{tick:.{decimals}f}" for tick in ticks]
        return tick_labels


class DurationFormatter(AxisFormatter):
    """Formatter for durations (values in seconds) with human-readable units.

    This formatter converts seconds to human-readable time units (s, min, h, d, mo, y)
    based on the range of values. It uses nice intervals (1, 2, 5, 10, 20, 50, etc.)
    for tick placement.
    """

    # Time unit conversions to seconds
    UNITS = [
        ("y", 365.25 * 24 * 3600),  # years (accounting for leap years)
        ("mo", 30.44 * 24 * 3600),  # months (average)
        ("d", 24 * 3600),  # days
        ("h", 3600),  # hours
        ("min", 60),  # minutes
        ("s", 1),  # seconds
    ]

    def _select_unit(self, min_: float, max_: float) -> tuple[str, float]:
        """Select the most appropriate time unit based on the range.

        Args:
            min_: Minimum value in seconds.
            max_: Maximum value in seconds.

        Returns:
            A tuple of (unit_name, unit_value_in_seconds).
        """
        range_seconds = max_ - min_
        # Select unit where the range would be reasonable (roughly 0.1 to 1000)
        for unit_name, unit_value in self.UNITS:
            if range_seconds >= unit_value * 0.5:
                return unit_name, unit_value
        # Default to seconds
        return "s", 1.0

    def get_ticks(self, min_: float, max_: float, max_ticks: int = 8) -> list[float]:
        """Generate tick positions at nice intervals.

        Args:
            min_: Minimum value in seconds.
            max_: Maximum value in seconds.
            max_ticks: Maximum number of ticks to generate. Defaults to 8.

        Returns:
            A list of tick positions in seconds.
        """
        # Select appropriate unit
        _, unit_value = self._select_unit(min_, max_)

        # Convert to the selected unit
        min_unit = min_ / unit_value
        max_unit = max_ / unit_value

        # Calculate nice intervals in the selected unit
        delta = max_unit - min_unit
        tick_spacing = delta / 5
        power = floor(log10(tick_spacing)) if tick_spacing > 0 else 0
        approx_interval = tick_spacing / 10**power
        intervals = np.array([1.0, 2.0, 5.0, 10.0])

        idx = intervals.searchsorted(approx_interval)
        interval = (intervals[idx - 1] if idx > 0 else intervals[0]) * 10**power
        if delta // interval > max_ticks:
            interval = intervals[idx] * 10**power

        # Generate ticks in the selected unit, then convert back to seconds
        ticks_in_unit = np.arange(
            ceil(min_unit / interval) * interval, max_unit + interval / 2, interval
        )
        ticks = [float(t * unit_value) for t in ticks_in_unit]
        return ticks

    def get_labels_for_ticks(self, ticks: Sequence[float]) -> list[str]:
        """Generate formatted labels for given tick positions.

        Args:
            ticks: A list of tick positions in seconds.

        Returns:
            A list of formatted tick labels with appropriate time units.
        """
        if not ticks:
            return []

        # Determine the unit based on the tick range
        min_tick = min(ticks)
        max_tick = max(ticks)
        unit_name, unit_value = self._select_unit(min_tick, max_tick)

        # Convert ticks to the selected unit
        ticks_in_unit = [t / unit_value for t in ticks]

        # Determine decimal places from tick spacing
        if len(ticks_in_unit) >= 2:
            spacing = ticks_in_unit[1] - ticks_in_unit[0]
            power = floor(log10(spacing)) if spacing > 0 else 0
        else:
            power = 0
        decimals = -min(0, power)

        # Format labels
        tick_labels = [f"{tick:.{decimals}f}{unit_name}" for tick in ticks_in_unit]
        return tick_labels
