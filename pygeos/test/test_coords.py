import pytest
import pygeos
from pygeos.ufuncs import count_coordinates, get_coordinates
import numpy as np
from numpy.testing import assert_equal

from .common import empty
from .common import point
from .common import point_z
from .common import line_string
from .common import linear_ring
from .common import polygon
from .common import polygon_with_hole
from .common import multi_point
from .common import multi_line_string
from .common import multi_polygon
from .common import geometry_collection

nested_2 = pygeos.geometrycollections([geometry_collection, point])
nested_3 = pygeos.geometrycollections([nested_2, point])


@pytest.mark.parametrize(
    "geoms,count",
    [
        ([], 0),
        ([empty], 0),
        ([point, empty], 1),
        ([empty, point, empty], 1),
        ([point, point], 2),
        ([point, point_z], 2),
        ([line_string, linear_ring], 8),
        ([polygon], 5),
        ([polygon_with_hole], 10),
        ([multi_point, multi_line_string], 4),
        ([multi_polygon], 10),
        ([geometry_collection], 3),
        ([nested_2], 4),
        ([nested_3], 5),
    ],
)
def test_count_coords(geoms, count):
    actual = count_coordinates(np.array(geoms, np.object))
    assert actual == count


# fmt: off
@pytest.mark.parametrize(
    "geoms,x,y",
    [
        ([], [], []),
        ([empty], [], []),
        ([point, empty], [2], [3]),
        ([empty, point, empty], [2], [3]),
        ([point, point], [2, 2], [3, 3]),
        ([point, point_z], [2, 1], [3, 1]),
        ([line_string, linear_ring], [0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0]),
        ([polygon], [0, 2, 2, 0, 0], [0, 0, 2, 2, 0]),
        ([polygon_with_hole], [0, 0, 10, 10, 0, 2, 2, 4, 4, 2], [0, 10, 10, 0, 0, 2, 4, 4, 2, 2]),
        ([multi_point, multi_line_string], [0, 1, 0, 1], [0, 2, 0, 2]),
        ([multi_polygon], [0, 1, 1, 0, 0, 2.1, 2.2, 2.2, 2.1, 2.1], [0, 0, 1, 1, 0, 2.1, 2.1, 2.2, 2.2, 2.1]),
        ([geometry_collection], [51, 52, 49], [-1, -1, 2]),
        ([nested_2], [51, 52, 49, 2], [-1, -1, 2, 3]),
        ([nested_3], [51, 52, 49, 2, 2], [-1, -1, 2, 3, 3]),
    ],
)  # fmt: on
def test_get_coords(geoms, x, y):
    actual = get_coordinates(np.array(geoms, np.object))
    expected = np.array([x, y], np.float64)
    assert_equal(actual, expected)
