import pytest

from textual_plot import AxisFormatter, NumericAxisFormatter


@pytest.fixture
def numeric_formatter() -> NumericAxisFormatter:
    return NumericAxisFormatter()


class TestNumericAxisFormatter:
    @pytest.mark.parametrize(
        "xmin, xmax, expected",
        [
            (0, 10, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
            (0, 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            (1 / 30, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0]),
        ],
    )
    def test_get_ticks(
        self, numeric_formatter: NumericAxisFormatter, xmin, xmax, expected
    ):
        ticks = numeric_formatter.get_ticks(xmin, xmax)
        assert ticks == pytest.approx(expected)

    @pytest.mark.parametrize(
        "xmin, xmax, expected",
        [
            (0, 10, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
            (0, 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            (1 / 30, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0]),
        ],
    )
    def test_get_ticks_and_labels(
        self, numeric_formatter: NumericAxisFormatter, xmin, xmax, expected
    ):
        ticks, labels = numeric_formatter.get_ticks_and_labels(xmin, xmax)
        assert ticks == pytest.approx(expected)
        # Ensure we get labels for each tick
        assert len(labels) == len(ticks)
        # Ensure labels are strings
        assert all(isinstance(label, str) for label in labels)

    def test_get_labels_for_ticks(self, numeric_formatter: NumericAxisFormatter):
        ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
        labels = numeric_formatter.get_labels_for_ticks(ticks)
        # With spacing of 0.5, decimals should be auto-determined as 1
        assert labels == ["0.0", "0.5", "1.0", "1.5", "2.0"]

    def test_get_labels_for_ticks_auto_decimals(
        self, numeric_formatter: NumericAxisFormatter
    ):
        # Test that decimals are automatically determined from tick spacing
        ticks = [0.0, 0.05, 0.10, 0.15, 0.20]
        labels = numeric_formatter.get_labels_for_ticks(ticks)
        # With spacing of 0.05, decimals should be 2
        assert labels == ["0.00", "0.05", "0.10", "0.15", "0.20"]

    def test_get_labels_for_empty_ticks(self, numeric_formatter: NumericAxisFormatter):
        labels = numeric_formatter.get_labels_for_ticks([])
        assert labels == []

    def test_is_axis_formatter(self, numeric_formatter: NumericAxisFormatter):
        # Ensure NumericAxisFormatter is an instance of AxisFormatter
        assert isinstance(numeric_formatter, AxisFormatter)
