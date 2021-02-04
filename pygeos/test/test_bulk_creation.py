import pygeos
import pytest
from .common import point, line_string

@pytest.mark.parametrize("geometries,indices,expected", [
    ([point, line_string], [0, 0], pygeos.geometrycollections([point, line_string]))
])
def test_collections_1d_scalar(geometries, indices, expected):
    assert pygeos.collections_1d(geometries, indices).equals(expected)
