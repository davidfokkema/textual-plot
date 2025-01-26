import pytest
from textual.geometry import Region

from textual_plot.plot_widget import map_coordinate_to_pixel


class TestImplementation:
    @pytest.mark.parametrize(
        "x, y, xmin, xmax, ymin, ymax, region,expected",
        [
            (0.0, 0.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (0, 3)),
            (1.0, 1.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (0, 3)),
            (4.99, 4.99, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (1, 3)),
            (1.0, 1.0, 0.0, 10.0, 0.0, 20.0, Region(2, 3, 4, 4), (2, 6)),
            (10.0, 20.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (4, -1)),
        ],
    )
    def test_map_coordinate_to_pixel(
        self, x, y, xmin, xmax, ymin, ymax, region, expected
    ):
        assert map_coordinate_to_pixel(x, y, xmin, xmax, ymin, ymax, region) == expected
