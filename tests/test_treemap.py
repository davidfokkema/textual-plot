"""Tests for treemap plot functionality."""

import numpy as np

from textual_plot.plot_widget import TreemapPlot, ValueDisplay


class TestTreemapPlot:
    """Test TreemapPlot dataclass and squarify integration."""

    def test_treemap_empty_after_filter_returns_early(self) -> None:
        """treemap() returns early when all values are NaN/Inf."""
        from textual_plot import PlotWidget

        plot = PlotWidget()
        plot.treemap([np.nan, np.inf])
        assert len(plot._datasets) == 0

    def test_squarify_layout_integration(self) -> None:
        """squarify produces valid rectangles for treemap layout."""
        import squarify

        values = [500, 433, 78, 25, 25, 7]
        width, height = 100, 50
        normalized = squarify.normalize_sizes(values, width, height)
        rects = squarify.squarify(normalized, 0, 0, width, height)
        assert len(rects) == len(values)
        total_area = sum(r["dx"] * r["dy"] for r in rects)
        assert abs(total_area - width * height) < 0.01  # Allow small float error

    def test_treemap_plot_structure(self) -> None:
        """TreemapPlot dataclass has expected structure for rendering."""
        values = np.array([10.0, 20.0, 30.0])
        dataset = TreemapPlot(
            values=values,
            labels=["A", "B", "C"],
            styles=["red", "blue", "green"],
            padding=1,
            hires_mode=None,
            aspect_preference=1.5,
            value_display=ValueDisplay.BOTH,
            currency_symbol="$",
        )
        assert len(dataset.values) == 3
        assert dataset.labels == ["A", "B", "C"]
        assert dataset.padding == 1
