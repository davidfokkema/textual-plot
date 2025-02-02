import pytest
from textual.geometry import Region

from textual_plot.plot_widget import (
    map_coordinate_to_hires_pixel,
    map_coordinate_to_pixel,
    map_pixel_to_coordinate,
)


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

    @pytest.mark.parametrize(
        "x, y, xmin, xmax, ymin, ymax, region,expected",
        [
            (0.0, 0.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (0.0, 4.0)),
            (1.0, 1.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (0.4, 3.8)),
            (5.0, 5.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (2.0, 3.0)),
            (1.0, 1.0, 0.0, 10.0, 0.0, 20.0, Region(2, 3, 4, 4), (2.4, 6.8)),
            (10.0, 20.0, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (4.0, 0.0)),
        ],
    )
    def test_map_coordinate_to_hires_pixel(
        self, x, y, xmin, xmax, ymin, ymax, region, expected
    ):
        assert (
            map_coordinate_to_hires_pixel(x, y, xmin, xmax, ymin, ymax, region)
            == expected
        )

    @pytest.mark.parametrize(
        "px, py, xmin, xmax, ymin, ymax, region,expected",
        [
            (0, 3, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (1.25, 2.5)),
            (2, 2, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (6.25, 7.5)),
            (1, 1, 0.0, 10.0, 0.0, 20.0, Region(0, 0, 4, 4), (3.75, 12.5)),
            (2, 3, 0.0, 10.0, 0.0, 20.0, Region(2, 3, 4, 4), (1.25, 17.5)),
        ],
    )
    def test_map_pixel_to_coordinate(
        self, px, py, xmin, xmax, ymin, ymax, region, expected
    ):
        assert (
            map_pixel_to_coordinate(px, py, xmin, xmax, ymin, ymax, region) == expected
        )
