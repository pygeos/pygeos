import pytest
import pygeos
from pygeos import Geometry

from .common import point, all_types

SET_OPERATIONS = (
    pygeos.difference,
    pygeos.intersection,
    pygeos.symmetric_difference,
    pygeos.union,
    # pygeos.coverage_union is tested seperately
)

REDUCE_SET_OPERATIONS = (
    (pygeos.intersection_all, pygeos.intersection),
    (pygeos.symmetric_difference_all, pygeos.symmetric_difference),
    (pygeos.union_all, pygeos.union),
    # (pygeos.coverage_union_all, pygeos.coverage_union) is tested seperately
)

reduce_test_data = [
    pygeos.box(0, 0, 5, 5),
    pygeos.box(2, 2, 7, 7),
    pygeos.box(4, 4, 9, 9),
]


@pytest.mark.parametrize("a", all_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_array(a, func):
    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("n", range(1, 4))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_1dim(n, func, related_func):
    actual = func(reduce_test_data[:n])
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i])
    assert pygeos.equals(actual, expected)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_axis(func, related_func):
    data = [[point] * 2] * 3  # shape = (3, 2)
    actual = func(data)
    assert actual.shape == (2,)
    actual = func(data, axis=0)  # default
    assert actual.shape == (2,)
    actual = func(data, axis=1)
    assert actual.shape == (3,)
    actual = func(data, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize("n", range(1, 4))
def test_coverage_union_reduce_1dim(n):
    """
    This is tested seperately from other set operations as it differs in two ways:
      1. It expects only non-overlapping polygons
      2. It expects GEOS 3.8.0+
    """
    test_data = [
        pygeos.box(0, 0, 1, 1),
        pygeos.box(1, 0, 2, 1),
        pygeos.box(2, 0, 3, 1),
    ]
    actual = pygeos.coverage_union_all(test_data[:n])
    # perform the reduction in a python loop and compare
    expected = test_data[0]
    for i in range(1, n):
        expected = pygeos.coverage_union(expected, test_data[i])
    assert pygeos.equals(actual, expected)
