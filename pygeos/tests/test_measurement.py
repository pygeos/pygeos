import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pygeos

from .common import (
    empty,
    geometry_collection,
    ignore_invalid,
    line_string,
    linear_ring,
    multi_line_string,
    multi_point,
    multi_polygon,
    point,
    point_polygon_testdata,
    polygon,
    polygon_with_hole,
)


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        geometry_collection,
    ],
)
def test_area_non_polygon(geom):
    assert pygeos.area(geom) == 0.0


def test_area():
    actual = pygeos.area([polygon, polygon_with_hole, multi_polygon])
    assert actual.tolist() == [4.0, 96.0, 1.01]


def test_distance():
    actual = pygeos.distance(*point_polygon_testdata)
    expected = [2 * 2 ** 0.5, 2 ** 0.5, 0, 0, 0, 2 ** 0.5]
    np.testing.assert_allclose(actual, expected)


def test_distance_missing():
    actual = pygeos.distance(point, None)
    assert np.isnan(actual)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, [2, 3, 2, 3]),
        ([point, multi_point], [[2, 3, 2, 3], [0, 0, 1, 2]]),
        (pygeos.linestrings([[0, 0], [0, 1]]), [0, 0, 0, 1]),
        (pygeos.linestrings([[0, 0], [1, 0]]), [0, 0, 1, 0]),
        (multi_point, [0, 0, 1, 2]),
        (multi_polygon, [0, 0, 2.2, 2.2]),
        (geometry_collection, [49, -1, 52, 2]),
        (empty, [np.nan, np.nan, np.nan, np.nan]),
        (None, [np.nan, np.nan, np.nan, np.nan]),
    ],
)
def test_bounds(geom, expected):
    assert_array_equal(pygeos.bounds(geom), expected)


@pytest.mark.parametrize(
    "geom,shape",
    [
        (point, (4,)),
        (None, (4,)),
        ([point, multi_point], (2, 4)),
        ([[point, multi_point], [polygon, point]], (2, 2, 4)),
        ([[[point, multi_point]], [[polygon, point]]], (2, 1, 2, 4)),
    ],
)
def test_bounds_dimensions(geom, shape):
    assert pygeos.bounds(geom).shape == shape


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, [2, 3, 2, 3]),
        (pygeos.linestrings([[0, 0], [0, 1]]), [0, 0, 0, 1]),
        (pygeos.linestrings([[0, 0], [1, 0]]), [0, 0, 1, 0]),
        (multi_point, [0, 0, 1, 2]),
        (multi_polygon, [0, 0, 2.2, 2.2]),
        (geometry_collection, [49, -1, 52, 2]),
        (empty, [np.nan, np.nan, np.nan, np.nan]),
        (None, [np.nan, np.nan, np.nan, np.nan]),
        ([empty, empty, None], [np.nan, np.nan, np.nan, np.nan]),
        # mixed missing and non-missing coordinates
        ([point, None], [2, 3, 2, 3]),
        ([point, empty], [2, 3, 2, 3]),
        ([point, empty, None], [2, 3, 2, 3]),
        ([point, empty, None, multi_point], [0, 0, 2, 3]),
    ],
)
def test_total_bounds(geom, expected):
    assert_array_equal(pygeos.total_bounds(geom), expected)


@pytest.mark.parametrize(
    "geom",
    [
        point,
        None,
        [point, multi_point],
        [[point, multi_point], [polygon, point]],
        [[[point, multi_point]], [[polygon, point]]],
    ],
)
def test_total_bounds_dimensions(geom):
    assert pygeos.total_bounds(geom).shape == (4,)


def test_length():
    actual = pygeos.length(
        [
            point,
            line_string,
            linear_ring,
            polygon,
            polygon_with_hole,
            multi_point,
            multi_polygon,
        ]
    )
    assert actual.tolist() == [0.0, 2.0, 4.0, 8.0, 48.0, 0.0, 4.4]


def test_length_missing():
    actual = pygeos.length(None)
    assert np.isnan(actual)


def test_hausdorff_distance():
    # example from GEOS docs
    a = pygeos.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = pygeos.linestrings([[0, 100], [0, 10], [80, 10]])
    with ignore_invalid():
        # Hausdorff distance emits "invalid value encountered"
        # (see https://github.com/libgeos/geos/issues/515)
        actual = pygeos.hausdorff_distance(a, b)
    assert actual == pytest.approx(22.360679775, abs=1e-7)


def test_hausdorff_distance_densify():
    # example from GEOS docs
    a = pygeos.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = pygeos.linestrings([[0, 100], [0, 10], [80, 10]])
    with ignore_invalid():
        # Hausdorff distance emits "invalid value encountered"
        # (see https://github.com/libgeos/geos/issues/515)
        actual = pygeos.hausdorff_distance(a, b, densify=0.001)
    assert actual == pytest.approx(47.8, abs=0.1)


def test_hausdorff_distance_missing():
    actual = pygeos.hausdorff_distance(point, None)
    assert np.isnan(actual)
    actual = pygeos.hausdorff_distance(point, None, densify=0.001)
    assert np.isnan(actual)


def test_hausdorff_densify_nan():
    actual = pygeos.hausdorff_distance(point, point, densify=np.nan)
    assert np.isnan(actual)


def test_distance_empty():
    actual = pygeos.distance(point, empty)
    assert np.isnan(actual)


def test_hausdorff_distance_empty():
    actual = pygeos.hausdorff_distance(point, empty)
    assert np.isnan(actual)


def test_hausdorff_distance_densify_empty():
    actual = pygeos.hausdorff_distance(point, empty, densify=0.2)
    assert np.isnan(actual)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize(
    "geom1, geom2, expected",
    [
        # identical geometries should have 0 distance
        (
            pygeos.linestrings([[0, 0], [100, 0]]),
            pygeos.linestrings([[0, 0], [100, 0]]),
            0,
        ),
        # example from GEOS docs
        (
            pygeos.linestrings([[0, 0], [50, 200], [100, 0], [150, 200], [200, 0]]),
            pygeos.linestrings([[0, 200], [200, 150], [0, 100], [200, 50], [0, 0]]),
            200,
        ),
        # same geometries but different curve direction results in maximum
        # distance between vertices on the lines.
        (
            pygeos.linestrings([[0, 0], [50, 200], [100, 0], [150, 200], [200, 0]]),
            pygeos.linestrings([[200, 0], [150, 200], [100, 0], [50, 200], [0, 0]]),
            200,
        ),
        # another example from GEOS docs
        (
            pygeos.linestrings([[0, 0], [50, 200], [100, 0], [150, 200], [200, 0]]),
            pygeos.linestrings([[0, 0], [200, 50], [0, 100], [200, 150], [0, 200]]),
            282.842712474619,
        ),
        # example from GEOS tests
        (
            pygeos.linestrings([[0, 0], [100, 0]]),
            pygeos.linestrings([[0, 0], [50, 50], [100, 0]]),
            70.7106781186548,
        ),
    ],
)
def test_frechet_distance(geom1, geom2, expected):
    actual = pygeos.frechet_distance(geom1, geom2)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize(
    "geom1, geom2, densify, expected",
    [
        # example from GEOS tests
        (
            pygeos.linestrings([[0, 0], [100, 0]]),
            pygeos.linestrings([[0, 0], [50, 50], [100, 0]]),
            0.001,
            50,
        )
    ],
)
def test_frechet_distance_densify(geom1, geom2, densify, expected):
    actual = pygeos.frechet_distance(geom1, geom2, densify=densify)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize(
    "geom1, geom2",
    [
        (line_string, None),
        (None, line_string),
        (None, None),
        (line_string, empty),
        (empty, line_string),
        (empty, empty),
    ],
)
def test_frechet_distance_nan_for_invalid_geometry_inputs(geom1, geom2):
    actual = pygeos.frechet_distance(geom1, geom2)
    assert np.isnan(actual)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_frechet_densify_ndarray():
    actual = pygeos.frechet_distance(
        pygeos.linestrings([[0, 0], [100, 0]]),
        pygeos.linestrings([[0, 0], [50, 50], [100, 0]]),
        densify=[0.1, 0.2, 1],
    )
    expected = np.array([50, 50.99019514, 70.7106781186548])
    np.testing.assert_array_almost_equal(actual, expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_frechet_densify_nan():
    actual = pygeos.frechet_distance(line_string, line_string, densify=np.nan)
    assert np.isnan(actual)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize("densify", [0, -1, 2])
def test_frechet_densify_invalid_values(densify):
    with pytest.raises(pygeos.GEOSException, match="Fraction is not in range"):
        pygeos.frechet_distance(line_string, line_string, densify=densify)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_frechet_distance_densify_empty():
    actual = pygeos.frechet_distance(line_string, empty, densify=0.2)
    assert np.isnan(actual)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_minimum_clearance():
    actual = pygeos.minimum_clearance([polygon, polygon_with_hole, multi_polygon])
    assert_allclose(actual, [2.0, 2.0, 0.1])


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_minimum_clearance_nonexistent():
    actual = pygeos.minimum_clearance([point, empty])
    assert np.isinf(actual).all()


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_minimum_clearance_missing():
    actual = pygeos.minimum_clearance(None)
    assert np.isnan(actual)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            pygeos.Geometry("POLYGON ((0 5, 5 10, 10 5, 5 0, 0 5))"),
            5,
        ),
        (
            pygeos.Geometry("LINESTRING (1 0, 1 10)"),
            5,
        ),
        (
            pygeos.Geometry("MULTIPOINT (2 2, 4 2)"),
            1,
        ),
        (
            pygeos.Geometry("POINT (2 2)"),
            0,
        ),
        (
            pygeos.Geometry("GEOMETRYCOLLECTION EMPTY"),
            0,
        ),
    ],
)
def test_minimum_bounding_radius(geometry, expected):
    actual = pygeos.minimum_bounding_radius(geometry)
    assert actual == pytest.approx(expected, abs=1e-12)
