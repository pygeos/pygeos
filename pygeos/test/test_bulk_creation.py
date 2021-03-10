import pygeos
import pytest
import numpy as np
from .common import point, line_string, linear_ring, polygon, empty

geom_coll = pygeos.geometrycollections


@pytest.mark.parametrize(
    "geometries",
    [
        np.array([1, 2], dtype=np.int32),
        None,
        np.array([[point]]),
        "hello",
    ],
)
def test_collections_1d_invalid_geometries(geometries):
    with pytest.raises(TypeError):
        pygeos.collections_1d(geometries, [0, 1])


@pytest.mark.parametrize(
    "indices",
    [
        np.array([point]),
        None,
        " hello",
        [0, 1],  # wrong length
    ],
)
def test_collections_1d_invalid_indices(indices):
    with pytest.raises((TypeError, ValueError)):
        pygeos.collections_1d([point], indices)


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
        ([point, None, line_string], [0, 0, 0], [geom_coll([point, line_string])])
    ],
)
def test_collections_1d(geometries, indices, expected):
    actual = pygeos.collections_1d(geometries, indices)
    assert pygeos.equals(actual, expected).all()


def test_collections_1d_multipoint():
    actual = pygeos.collections_1d(
        [point], [0], geometry_type=pygeos.GeometryType.MULTIPOINT
    )
    assert pygeos.equals(actual, pygeos.multipoints([point])).all()


def test_collections_1d_multilinestring():
    actual = pygeos.collections_1d(
        [line_string], [0], geometry_type=pygeos.GeometryType.MULTILINESTRING
    )
    assert pygeos.equals(actual, pygeos.multilinestrings([line_string])).all()


def test_collections_1d_multilinearring():
    actual = pygeos.collections_1d(
        [linear_ring], [0], geometry_type=pygeos.GeometryType.MULTILINESTRING
    )
    assert pygeos.equals(actual, pygeos.multilinestrings([linear_ring])).all()


def test_collections_1d_multipolygon():
    actual = pygeos.collections_1d(
        [polygon], [0], geometry_type=pygeos.GeometryType.MULTIPOLYGON
    )
    assert pygeos.equals(actual, pygeos.multipolygons([polygon])).all()


@pytest.mark.parametrize(
    "geometries,geom_type",
    [
        ([line_string], pygeos.GeometryType.MULTIPOINT),
        ([polygon], pygeos.GeometryType.MULTIPOINT),
        ([point], pygeos.GeometryType.MULTILINESTRING),
        ([polygon], pygeos.GeometryType.MULTILINESTRING),
        ([point], pygeos.GeometryType.MULTIPOLYGON),
        ([line_string], pygeos.GeometryType.MULTIPOLYGON),
    ],
)
def test_collections_1d_incompatible_types(geometries, geom_type):
    with pytest.raises(TypeError):
        pygeos.collections_1d(geometries, [0], geom_type)
