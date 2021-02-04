import pygeos
import pytest
import numpy as np
from .common import point, line_string, empty

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
    ],
)
def test_collections_1d(geometries, indices, expected):
    actual = pygeos.collections_1d(geometries, indices)
    assert pygeos.equals(actual, expected).all()
