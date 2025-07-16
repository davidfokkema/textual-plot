import pytest

from textual_plot import PlotWidget


@pytest.fixture
def plot() -> PlotWidget:
    return PlotWidget()


class TestTicks:
    @pytest.mark.parametrize(
        "xmin, xmax, expected",
        [
            (0, 10, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
            (0, 1, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            (1 / 30, 1.0, [0.2, 0.4, 0.6, 0.8, 1.0]),
        ],
    )
    def test_ticks_from_get_ticks_between(self, plot: PlotWidget, xmin, xmax, expected):
        ticks, _ = plot.get_ticks_between(xmin, xmax)
        assert ticks == pytest.approx(expected)
