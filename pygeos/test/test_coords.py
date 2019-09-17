import pytest
import pygeos
import numpy as np
from numpy.testing import assert_equal

from .common import all_types, empty


@pytest.fixture
def geometries():
    n = 3  # number of geometries per type
    m_linestring = 5  # coords in linestring
    n_coords = 3 + 3 * 5
    coords = np.random.random((n_coords, 2))
    points = pygeos.points(coords[:3])
    linestrings = pygeos.linestrings(coords[3:].reshape(n, m_linestring, 2))
    return np.concatenate([points, linestrings]), coords.T


def test_count_coords_empty(geometries):
    assert pygeos.ufuncs.count_coordinates(np.array([empty, empty])) == 0


def test_get_coords_empty(geometries):
    actual = pygeos.ufuncs.get_coordinates(np.array([empty, empty]))
    assert actual.shape == (2, 0)


def test_count_coords_simple(geometries):
    assert pygeos.ufuncs.count_coordinates(geometries[0]) == 3 + 3 * 5


def test_get_coords_simple(geometries):
    actual = pygeos.ufuncs.get_coordinates(geometries[0])
    assert_equal(actual, geometries[1])


def test_count_coords_all_types():
    expected = 31
    assert pygeos.ufuncs.count_coordinates(np.array(all_types)) == expected


def test_get_coords_all_types():
    expected = (2, 31)
    actual = pygeos.ufuncs.get_coordinates(np.array(all_types))
    assert actual.shape == expected
