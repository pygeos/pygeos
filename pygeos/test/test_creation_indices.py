import pygeos
import pytest
import numpy as np
from .common import point, line_string, linear_ring, polygon, empty, empty_point

pnts = pygeos.points
lstrs = pygeos.linestrings
geom_coll = pygeos.geometrycollections
lstr_empty = pygeos.Geometry("LINESTRING EMPTY")


@pytest.mark.parametrize(
    "func", [pygeos.points, pygeos.linestrings, pygeos.linearrings]
)
@pytest.mark.parametrize(
    "coordinates",
    [
        np.empty((2,)),  # not enough dimensions
        np.empty((2, 4, 1)),  # too many dimensions
        np.empty((2, 4)),  # wrong inner dimension size
        None,
        np.full((2, 2), "foo"),  # wrong type
    ],
)
def test_invalid_coordinates(func, coordinates):
    with pytest.raises((TypeError, ValueError)):
        func(coordinates, indices=[0, 1])


@pytest.mark.parametrize(
    "func",
    [
        pygeos.multipoints,
        pygeos.multilinestrings,
        pygeos.multipolygons,
        pygeos.geometrycollections,
    ],
)
@pytest.mark.parametrize(
    "geometries", [np.array([1, 2], dtype=np.int32), None, np.array([[point]]), "hello"]
)
def test_invalid_geometries(func, geometries):
    with pytest.raises((TypeError, ValueError)):
        func(geometries, indices=[0, 1])


@pytest.mark.parametrize(
    "func", [pygeos.points, pygeos.linestrings, pygeos.linearrings]
)
@pytest.mark.parametrize(
    "indices", [np.array([point]), " hello", [0, 1]]  # wrong length
)
def test_invalid_indices_simple(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([[0.2, 0.3]], indices=indices)


def test_points_invalid():
    with pytest.raises(pygeos.GEOSException):
        pygeos.points([[1, 1], [2, 2]], indices=[0, 0])


def test_points():
    actual = pygeos.points([[2, 3], [2, 3]], indices=[0, 2])
    assert pygeos.equals(actual, [point, empty_point, point]).all()


@pytest.mark.parametrize(
    "coordinates,indices,expected",
    [
        ([[1, 1], [2, 2]], [0, 0], [lstrs([[1, 1], [2, 2]])]),
        ([[1, 1, 1], [2, 2, 2]], [0, 0], [lstrs([[1, 1, 1], [2, 2, 2]])]),
        ([[1, 1], [2, 2]], [1, 1], [lstr_empty, lstrs([[1, 1], [2, 2]])]),
        (
            [[1, 1], [2, 2], [2, 2], [3, 3]],
            [0, 0, 1, 1],
            [lstrs([[1, 1], [2, 2]]), lstrs([[2, 2], [3, 3]])],
        ),
    ],
)
def test_linestrings(coordinates, indices, expected):
    actual = pygeos.linestrings(coordinates, indices=indices)
    assert pygeos.equals(actual, expected).all()


def test_linestrings_invalid():
    with pytest.raises(pygeos.GEOSException):
        pygeos.linestrings([[1, 1], [2, 2]], indices=[0, 1])


@pytest.mark.parametrize(
    "coordinates", [([[1, 1], [2, 1], [2, 2], [1, 1]]), ([[1, 1], [2, 1], [2, 2]])]
)
def test_linearrings(coordinates):
    actual = pygeos.linearrings(coordinates, indices=len(coordinates) * [0])
    assert pygeos.equals(actual, pygeos.linearrings(coordinates)).all()


def test_linearrings_invalid():
    with pytest.raises(pygeos.GEOSException):
        pygeos.linearrings([[1, 1], [2, 1], [1, 1]], indices=[0, 0, 0])


@pytest.mark.parametrize(
    "func",
    [
        pygeos.multipoints,
        pygeos.multilinestrings,
        pygeos.multipolygons,
        pygeos.geometrycollections,
    ],
)
@pytest.mark.parametrize(
    "indices", [np.array([point]), " hello", [0, 1]]  # wrong length
)
def test_invalid_indices_collections(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([point], indices=indices)


@pytest.mark.parametrize(
    "geometries,indices,expected",
    [
        ([point, line_string], [0, 0], [geom_coll([point, line_string])]),
        ([point, line_string], [0, 1], [geom_coll([point]), geom_coll([line_string])]),
        (
            [point, line_string],
            [1, 1],
            [geom_coll([]), geom_coll([point, line_string])],
        ),
        ([point, None], [0, 0], [geom_coll([point])]),
        ([point, None], [0, 1], [geom_coll([point]), geom_coll([])]),
        ([point, None, line_string], [0, 0, 0], [geom_coll([point, line_string])]),
    ],
)
def test_geometrycollections(geometries, indices, expected):
    actual = pygeos.geometrycollections(geometries, indices=indices)
    assert pygeos.equals(actual, expected).all()


def test_multipoints():
    actual = pygeos.multipoints([point], indices=[0])
    assert pygeos.equals(actual, pygeos.multipoints([point])).all()


def test_multilinestrings():
    actual = pygeos.multilinestrings([line_string], indices=[0])
    assert pygeos.equals(actual, pygeos.multilinestrings([line_string])).all()


def test_multilinearrings():
    actual = pygeos.multilinestrings([linear_ring], indices=[0])
    assert pygeos.equals(actual, pygeos.multilinestrings([linear_ring])).all()


def test_multipolygons():
    actual = pygeos.multipolygons([polygon], indices=[0])
    assert pygeos.equals(actual, pygeos.multipolygons([polygon])).all()


@pytest.mark.parametrize(
    "geometries,func",
    [
        ([line_string], pygeos.multipoints),
        ([polygon], pygeos.multipoints),
        ([point], pygeos.multilinestrings),
        ([polygon], pygeos.multilinestrings),
        ([point], pygeos.multipolygons),
        ([line_string], pygeos.multipolygons),
    ],
)
def test_incompatible_types(geometries, func):
    with pytest.raises(TypeError):
        func(geometries, indices=[0])
