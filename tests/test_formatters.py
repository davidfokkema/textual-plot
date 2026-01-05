import pytest

from textual_plot import AxisFormatter, DurationFormatter, NumericAxisFormatter


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


@pytest.fixture
def duration_formatter() -> DurationFormatter:
    return DurationFormatter()


class TestDurationFormatter:
    @pytest.mark.parametrize(
        "xmin, xmax, expected_unit",
        [
            (0, 10, "s"),  # 10 seconds
            (0, 300, "min"),  # 5 minutes
            (0, 7200, "h"),  # 2 hours
            (0, 172800, "d"),  # 2 days
            (0, 5184000, "mo"),  # ~2 months
            (0, 63072000, "y"),  # ~2 years
        ],
    )
    def test_unit_selection(
        self, duration_formatter: DurationFormatter, xmin, xmax, expected_unit
    ):
        ticks, labels = duration_formatter.get_ticks_and_labels(xmin, xmax)
        # Check that labels contain the expected unit
        assert all(expected_unit in label for label in labels)

    def test_get_ticks_seconds(self, duration_formatter: DurationFormatter):
        # Test for a range in seconds (0-10 seconds)
        ticks = duration_formatter.get_ticks(0, 10)
        # Should get nice intervals like 0, 2, 4, 6, 8, 10
        assert ticks == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    def test_get_ticks_minutes(self, duration_formatter: DurationFormatter):
        # Test for a range in minutes (0-600 seconds = 10 minutes)
        ticks = duration_formatter.get_ticks(0, 600)
        # Should get ticks in minute intervals (0, 2, 4, 6, 8, 10 minutes)
        assert ticks == pytest.approx([0.0, 120.0, 240.0, 360.0, 480.0, 600.0])

    def test_get_ticks_hours(self, duration_formatter: DurationFormatter):
        # Test for a range in hours (0-7200 seconds = 2 hours)
        ticks = duration_formatter.get_ticks(0, 7200)
        # Should get ticks at 0.5 hour intervals
        assert ticks == pytest.approx([0.0, 1800.0, 3600.0, 5400.0, 7200.0])

    def test_get_labels_for_ticks(self, duration_formatter: DurationFormatter):
        # Test label formatting for seconds
        ticks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        labels = duration_formatter.get_labels_for_ticks(ticks)
        assert labels == ["0s", "2s", "4s", "6s", "8s", "10s"]

    def test_get_labels_for_ticks_minutes(self, duration_formatter: DurationFormatter):
        # Test label formatting for minutes
        ticks = [0.0, 120.0, 240.0, 360.0, 480.0, 600.0]  # 0, 2, 4, 6, 8, 10 minutes
        labels = duration_formatter.get_labels_for_ticks(ticks)
        assert labels == ["0min", "2min", "4min", "6min", "8min", "10min"]

    def test_get_labels_for_ticks_hours(self, duration_formatter: DurationFormatter):
        # Test label formatting for hours
        ticks = [0.0, 3600.0, 7200.0]  # 0, 1, 2 hours
        labels = duration_formatter.get_labels_for_ticks(ticks)
        assert labels == ["0h", "1h", "2h"]

    def test_get_labels_for_empty_ticks(self, duration_formatter: DurationFormatter):
        labels = duration_formatter.get_labels_for_ticks([])
        assert labels == []

    def test_get_ticks_and_labels(self, duration_formatter: DurationFormatter):
        ticks, labels = duration_formatter.get_ticks_and_labels(0, 10)
        # Ensure we get labels for each tick
        assert len(labels) == len(ticks)
        # Ensure labels are strings
        assert all(isinstance(label, str) for label in labels)
        # Ensure labels contain a unit
        assert all(
            any(unit in label for unit, _ in DurationFormatter.UNITS)
            for label in labels
        )

    def test_is_axis_formatter(self, duration_formatter: DurationFormatter):
        # Ensure DurationFormatter is an instance of AxisFormatter
        assert isinstance(duration_formatter, AxisFormatter)
