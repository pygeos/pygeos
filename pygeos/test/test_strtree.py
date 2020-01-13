import math
import pygeos
from pygeos import box
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from .common import point, empty, assert_increases_refcount, assert_decreases_refcount


# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9


@pytest.fixture
def tree():
    geoms = pygeos.points(np.arange(10), np.arange(10))
    yield pygeos.STRtree(geoms)


@pytest.fixture
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = pygeos.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield pygeos.STRtree(geoms)


@pytest.fixture
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = pygeos.buffer(
        pygeos.points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quadsegs=64
    )
    yield pygeos.STRtree(geoms)


def test_init_with_none():
    tree = pygeos.STRtree(np.array([None]))
    assert tree.query(point).size == 0


def test_init_with_no_geometry():
    with pytest.raises(TypeError):
        pygeos.STRtree(np.array(["Not a geometry"], dtype=object))


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(arr):
        _ = pygeos.STRtree(arr)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    with assert_decreases_refcount(arr):
        del tree


def test_geometries_property():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    assert arr is tree.geometries


def test_flush_geometries(tree):
    arr = pygeos.points(np.arange(10), np.arange(10))
    tree = pygeos.STRtree(arr)
    # Dereference geometries
    arr[:] = None
    # Still it does not lead to a segfault
    tree.query(point)


def test_query_no_geom(tree):
    with pytest.raises(TypeError):
        tree.query("I am not a geometry")


def test_query_none(tree):
    assert tree.query(None).size == 0


def test_query_empty(tree):
    assert tree.query(empty).size == 0


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box contains points
        (box(0, 0, 1, 1), [0, 1]),
        # box contains points
        (box(5, 5, 15, 15), [5, 6, 7, 8, 9]),
        # envelope of buffer contains points
        (pygeos.buffer(pygeos.points(3, 3), 1), [2, 3, 4]),
        # envelope of points contains points
        (pygeos.multipoints([[5, 7], [7, 5]]), [5, 6, 7]),
    ],
)
def test_query_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (pygeos.points(0, 0), [0]),
        (pygeos.points(0.5, 0.5), [0]),
        # point within envelope of first line
        (pygeos.points(0, 0.5), [0]),
        # point at shared vertex between 2 lines
        (pygeos.points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        # envelope of buffer overlaps envelope of 2 lines
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [2, 3]),
        # envelope of points overlaps 5 lines (touches edge of 2 envelopes)
        (pygeos.multipoints([[5, 7], [7, 5]]), [4, 5, 6, 7]),
    ],
)
def test_query_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects edge of envelopes of 2 polygons
        (pygeos.points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (pygeos.points(1, 1), [1]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box overlaps envelope of 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        # envelope of buffer overlaps envelope of 3 polygons
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # envelope of larger buffer overlaps envelope of 6 polygons
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        # envelope of points overlaps 3 polygons
        (pygeos.multipoints([[5, 7], [7, 5]]), [5, 6, 7]),
    ],
)
def test_query_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry), expected)


def test_query_invalid_predicate(tree):
    with pytest.raises(ValueError):
        tree.query(pygeos.points(1, 1), predicate="bad_predicate")


def test_query_unsupported_predicate(tree):
    # valid GEOS binary predicate, but not supported for query
    with pytest.raises(ValueError):
        tree.query(pygeos.points(1, 1), predicate="disjoint")


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box contains points
        (box(3, 3, 6, 6), [3, 4, 5, 6]),
        # envelope of buffer contains more points than intersect buffer
        # due to diagonal distance
        (pygeos.buffer(pygeos.points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect
        (pygeos.multipoints([[5, 5], [7, 7]]), [5, 7]),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_intersects_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (pygeos.points(0, 0), [0]),
        (pygeos.points(0.5, 0.5), [0]),
        # point within envelope of first line but does not intersect
        (pygeos.points(0, 0.5), []),
        # point at shared vertex between 2 lines
        (pygeos.points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        # buffer intersects 2 lines
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [2, 3]),
        # buffer intersects midpoint of line at tangent
        (pygeos.buffer(pygeos.points(2, 1), HALF_UNIT_DIAG), [1]),
        # envelope of points overlaps 5 lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), [6, 7]),
    ],
)
def test_query_intersects_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (pygeos.points(0, 0.5), [0]),
        (pygeos.points(0.5, 0), [0]),
        # midpoint between two polygons intersects both
        (pygeos.points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (pygeos.points(1, 1), [1]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box intersects 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        # buffer overlaps 3 polygons
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # larger buffer overlaps 6 polygons (touches midpoints)
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        # envelope of points overlaps polygons, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects at vertex of 2 lines
        (pygeos.multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_intersects_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="intersects"), expected)
