import pygeos
import numpy as np
import pytest

from pygeos import Geometry, GEOSException

from .common import point, line_string, all_types, empty

CONSTRUCTIVE_NO_ARGS = (
    pygeos.boundary,
    pygeos.centroid,
    pygeos.convex_hull,
    pygeos.envelope,
    pygeos.extract_unique_points,
    pygeos.normalize,
    pygeos.point_on_surface,
)

CONSTRUCTIVE_FLOAT_ARG = (
    pygeos.buffer,
    pygeos.delaunay_triangles,
    pygeos.simplify,
    pygeos.voronoi_polygons,
)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_array(geometry, func):
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_array(geometry, func):
    actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("reference", all_types)
def test_snap_array(geometry, reference):
    actual = pygeos.snap([geometry, geometry], [reference, reference], tolerance=1.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_missing(func):
    actual = func(None)
    assert actual is None


@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_missing(func):
    actual = func(None, 1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_nan(geometry, func):
    actual = func(geometry, float("nan"))
    assert actual is None


def test_snap_none():
    actual = pygeos.snap(None, point, tolerance=1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
def test_snap_nan_float(geometry):
    actual = pygeos.snap(geometry, point, tolerance=np.nan)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_build_area_none():
    actual = pygeos.build_area(None)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, empty),  # a point has no area
        (line_string, empty),  # a line string has no area
        # geometry collection of two polygons are combined into one
        (
            Geometry("GEOMETRYCOLLECTION(POLYGON((0 0, 3 0, 3 3, 0 3, 0 0)), POLYGON((1 1, 1 2, 2 2, 1 1)))"),
            Geometry("POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 2 2, 1 2, 1 1))"),
        ),
        (empty, empty),
        ([empty], [empty])
    ],
)
def test_build_area(geom, expected):
    actual = pygeos.build_area(geom)
    assert actual is not expected
    assert actual == expected


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_make_valid_none():
    actual = pygeos.make_valid(None)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a valid geometry stays the same (but is copied)
        # an L shaped polygon without area is converted to a multilinestring
        (
            Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"),
            Geometry("MULTILINESTRING ((0 0, 1 1), (1 1, 1 2))"),
        ),
        # a polygon with self-intersection (bowtie) is converted into polygons
        (
            Geometry("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))"),
            Geometry("MULTIPOLYGON (((1 1, 0 0, 0 2, 1 1)), ((1 1, 2 2, 2 0, 1 1)))"),
        ),
        (empty, empty),
        ([empty], [empty])
    ],
)
def test_make_valid(geom, expected):
    actual = pygeos.make_valid(geom)
    assert actual is not expected
    assert actual == expected

@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (all_types, all_types),
        # first polygon is valid, second polygon has self-intersection
        (
            [
                Geometry("POLYGON((0 0, 2 2, 0 2, 0 0))"),
                Geometry("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))"),
            ],
            [
                Geometry("POLYGON((0 0, 2 2, 0 2, 0 0))"),
                Geometry(
                    "MULTIPOLYGON (((1 1, 0 0, 0 2, 1 1)), ((1 1, 2 2, 2 0, 1 1)))"
                ),
            ],
        ),
        ([point, None, empty], [point, None, empty])
    ],
)
def test_make_valid_1d(geom, expected):
    actual = pygeos.make_valid(geom)
    assert np.all(actual == expected)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a point is always in normalized form
        # order coordinates of linestrings and parts of multi-linestring
        (
            Geometry("MULTILINESTRING ((1 1, 0 0), (1 1, 1 2))"),
            Geometry("MULTILINESTRING ((1 1, 1 2), (0 0, 1 1))"),
        ),
    ],
)
def test_normalize(geom, expected):
    actual = pygeos.normalize(geom)
    assert actual == expected
