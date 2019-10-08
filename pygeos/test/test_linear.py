import pygeos
import numpy as np

from .common import line_string
from .common import linear_ring
from .common import multi_line_string

pygeos.linestrings([(0, 0), (1, 0), (1, 1)])


def test_line_interpolate_point_geom_array():
    actual = pygeos.line_interpolate_point([line_string, linear_ring], -1)
    assert pygeos.equals(actual[0], pygeos.Geometry("POINT (1 0)"))
    assert pygeos.equals(actual[1], pygeos.Geometry("POINT (0 1)"))


def test_line_interpolate_point_float_array():
    actual = pygeos.line_interpolate_point(line_string, [0.2, 1.5, -0.2])
    assert pygeos.equals(actual[0], pygeos.Geometry("POINT (0.2 0)"))
    assert pygeos.equals(actual[1], pygeos.Geometry("POINT (1 0.5)"))
    assert pygeos.equals(actual[2], pygeos.Geometry("POINT (1 0.8)"))


def test_line_interpolate_point_none():
    assert pygeos.line_interpolate_point(None, 0.2) is None


def test_line_interpolate_point_nan():
    assert pygeos.line_interpolate_point(line_string, np.nan) is None


def test_line_locate_point_geom_array():
    point = pygeos.points(0, 1)
    actual = pygeos.line_locate_point([line_string, linear_ring], point)
    np.testing.assert_allclose(actual, [0.0, 3.0])


def test_line_locate_point_geom_array2():
    points = pygeos.points([[0, 0], [1, 0]])
    actual = pygeos.line_locate_point(line_string, points)
    np.testing.assert_allclose(actual, [0.0, 1.0])


def test_line_locate_point_none():
    assert np.isnan(pygeos.line_locate_point(line_string, None))
    assert np.isnan(pygeos.line_locate_point(None, pygeos.points(0, 0)))


def test_line_merge_geom_array():
    actual = pygeos.line_merge([line_string, multi_line_string])
    assert pygeos.equals(actual[0], line_string)
    assert pygeos.equals(actual[1], multi_line_string)
