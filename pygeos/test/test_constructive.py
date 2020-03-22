import pygeos
import numpy as np
import pytest

from pygeos import Geometry, GEOSException

from .common import point, all_types

CONSTRUCTIVE_NO_ARGS = (
    pygeos.boundary,
    pygeos.centroid,
    pygeos.convex_hull,
    pygeos.envelope,
    pygeos.extract_unique_points,
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
def test_make_valid_none():
    actual = pygeos.make_valid(None)
    assert actual is None



@pytest.mark.skipif(pygeos.geos_version >= (3, 8, 0), reason="GEOS >= 3.8")
def test_make_valid_importerror():
    with pytest.raises(ImportError):
        pygeos.make_valid(None)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize("geom,expected", [
    ("POINT(0 0)", "POINT(0 0)"),
    ("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))", "POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))")
])
def test_make_valid(geom, expected):
    actual = pygeos.make_valid("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))")
    assert pygeos.to_wkt(actual) == expected
