import pytest
import pygeos
import numpy as np
from numpy.testing import assert_equal

from .common import point, line_string


@pytest.fixture
def geometries():
    return np.array([point, line_string])


def test_count_coords(geometries):
    assert pygeos.ufuncs.count_coordinates(geometries) == 4


def test_get_coords(geometries):
    expected = np.array([[2, 0, 1, 1], [3, 0, 0, 1]], dtype=np.float64)
    actual = pygeos.ufuncs.get_coordinates(geometries)
    assert_equal(actual, expected)
