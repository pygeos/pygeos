import pytest

from pygeos.testing import assert_geometries_equal

from .common import (
    empty_line_string,
    empty_line_string_z,
    empty_point,
    empty_point_z,
    empty_polygon,
    line_string_nan,
)


@pytest.mark.parametrize(
    "geom",
    [
        empty_point,
        empty_point_z,
        empty_line_string,
        empty_line_string_z,
        empty_polygon,
        line_string_nan,
        None,
    ],
)
def test_equals_exact_true_empty_and_nan(geom):
    assert_geometries_equal(geom, geom)


@pytest.mark.parametrize(
    "geom1",
    [
        empty_point,
        empty_point_z,
        empty_line_string,
        empty_line_string_z,
        empty_polygon,
        line_string_nan,
        None,
    ],
)
@pytest.mark.parametrize(
    "geom2",
    [
        empty_point,
        empty_point_z,
        empty_line_string,
        empty_line_string_z,
        empty_polygon,
        line_string_nan,
        None,
    ],
)
def test_equals_exact_false_empty_and_nan(geom1, geom2):
    if geom1 is geom2:
        pytest.skip()
    with pytest.raises(AssertionError):
        assert_geometries_equal(geom1, geom2)
