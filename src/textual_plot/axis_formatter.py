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
