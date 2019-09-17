import pytest
import pygeos
import numpy as np
from numpy.testing import assert_equal

from .common import point, line_string


@pytest.fixture
def geometries():
    n = 3  # number of geometries per type
    m_linestring = 5  # coords in linestring
    n_coords = 3 + 3 * 5
    coords = np.random.random((n_coords, 2))
    points = pygeos.points(coords[:3])
    linestrings = pygeos.linestrings(coords[3:].reshape(n, m_linestring, 2))
    return np.concatenate([points, linestrings]), coords.T


def test_count_coords(geometries):
    assert pygeos.ufuncs.count_coordinates(geometries[0]) == 3 + 3 * 5


def test_get_coords(geometries):
    actual = pygeos.ufuncs.get_coordinates(geometries[0])
    assert_equal(actual, geometries[1])
