import math
import pygeos
from pygeos import box, points, linestrings, multipoints, buffer, STRtree
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from .common import point, empty, assert_increases_refcount, assert_decreases_refcount


# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9


@pytest.fixture
def tree():
    geoms = points(np.arange(10), np.arange(10))
    yield STRtree(geoms)


@pytest.fixture
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield STRtree(geoms)


@pytest.fixture
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = buffer(
        points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quadsegs=32
    )
    yield STRtree(geoms)


def test_init_with_none():
    tree = STRtree(np.array([None]))
    assert tree.query(point).size == 0


def test_init_with_no_geometry():
    with pytest.raises(TypeError):
        STRtree(np.array(["Not a geometry"], dtype=object))


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(point):
        _ = STRtree(arr)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = STRtree(arr)
    with assert_decreases_refcount(point):
        del tree


def test_flush_geometries():
    arr = points(np.arange(10), np.arange(10))
    tree = STRtree(arr)
    # Dereference geometries
    arr[:] = None
    import gc

    gc.collect()
    # Still it does not lead to a segfault
    tree.query(point)


def test_len():
    arr = np.array([point, None, point])
    tree = STRtree(arr)
    assert len(tree) == 2


def test_geometries_property():
    arr = np.array([point])
    tree = STRtree(arr)
    assert arr is tree.geometries


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
        (points(0.5, 0.5), []),
        # points intersect
        (points(1, 1), [1]),
        # box contains points
        (box(0, 0, 1, 1), [0, 1]),
        # box contains points
        (box(5, 5, 15, 15), [5, 6, 7, 8, 9]),
        # envelope of buffer contains points
        (buffer(points(3, 3), 1), [2, 3, 4]),
        # envelope of points contains points
        (multipoints([[5, 7], [7, 5]]), [5, 6, 7]),
    ],
)
def test_query_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (points(0, 0), [0]),
        (points(0.5, 0.5), [0]),
        # point within envelope of first line
        (points(0, 0.5), [0]),
        # point at shared vertex between 2 lines
        (points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        # envelope of buffer overlaps envelope of 2 lines
        (buffer(points(3, 3), 0.5), [2, 3]),
        # envelope of points overlaps 5 lines (touches edge of 2 envelopes)
        (multipoints([[5, 7], [7, 5]]), [4, 5, 6, 7]),
    ],
)
def test_query_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects edge of envelopes of 2 polygons
        (points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (points(1, 1), [1]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box overlaps envelope of 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        # envelope of buffer overlaps envelope of 3 polygons
        (buffer(points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # envelope of larger buffer overlaps envelope of 6 polygons
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        # envelope of points overlaps 3 polygons
        (multipoints([[5, 7], [7, 5]]), [5, 6, 7]),
    ],
)
def test_query_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry), expected)


def test_query_invalid_predicate(tree):
    with pytest.raises(ValueError):
        tree.query(points(1, 1), predicate="bad_predicate")


def test_query_unsupported_predicate(tree):
    # valid GEOS binary predicate, but not supported for query
    with pytest.raises(ValueError):
        tree.query(points(1, 1), predicate="disjoint")


def test_query_tree_with_none():
    # valid GEOS binary predicate, but not supported for query
    tree = STRtree(
        [pygeos.Geometry("POINT (0 0)"), None, pygeos.Geometry("POINT (2 2)")]
    )
    assert tree.query(points(2, 2), predicate="intersects") == [2]


### predicate == 'intersects'

# TEMPORARY xfail: MultiPoint intersects with prepared geometries does not work
# properly on GEOS 3.5.x; it was fixed in 3.6+
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (points(0.5, 0.5), []),
        # points intersect
        (points(1, 1), [1]),
        # box contains points
        (box(3, 3, 6, 6), [3, 4, 5, 6]),
        # envelope of buffer contains more points than intersect buffer
        # due to diagonal distance
        (buffer(points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect
        pytest.param(
            multipoints([[5, 5], [7, 7]]),
            [5, 7],
            marks=pytest.mark.xfail(pygeos.geos_version < (3, 6, 0), reason="GEOS 3.5"),
        ),
        # envelope of points contains points, but points do not intersect
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        pytest.param(
            multipoints([[5, 7], [7, 7]]),
            [7],
            marks=pytest.mark.xfail(pygeos.geos_version < (3, 6, 0), reason="GEOS 3.5"),
        ),
    ],
)
def test_query_intersects_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (points(0, 0), [0]),
        (points(0.5, 0.5), [0]),
        # point within envelope of first line but does not intersect
        (points(0, 0.5), []),
        # point at shared vertex between 2 lines
        (points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        # buffer intersects 2 lines
        (buffer(points(3, 3), 0.5), [2, 3]),
        # buffer intersects midpoint of line at tangent
        (buffer(points(2, 1), HALF_UNIT_DIAG), [1]),
        # envelope of points overlaps lines but intersects none
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (multipoints([[5, 7], [7, 7]]), [6, 7]),
    ],
)
def test_query_intersects_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (points(0, 0.5), [0]),
        (points(0.5, 0), [0]),
        # midpoint between two polygons intersects both
        (points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (points(1, 1), [1]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box intersects 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        # buffer overlaps 3 polygons
        (buffer(points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # larger buffer overlaps 6 polygons (touches midpoints)
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        # envelope of points overlaps polygons, but points do not intersect
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint within polygon
        (multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_intersects_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="intersects"), expected)


### predicate == 'within'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (points(0.5, 0.5), []),
        # points intersect
        (points(1, 1), [1]),
        # box not within points
        (box(3, 3, 6, 6), []),
        # envelope of buffer not within points
        (buffer(points(3, 3), 1), []),
        # multipoints intersect but are not within points in tree
        (multipoints([[5, 5], [7, 7]]), []),
        # only one point of multipoint intersects, but multipoints are not
        # within any points in tree
        (multipoints([[5, 7], [7, 7]]), []),
        # envelope of points contains points, but points do not intersect
        (multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_within_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="within"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # endpoint not within first line
        (points(0, 0), []),
        # point within first line
        (points(0.5, 0.5), [0]),
        # point within envelope of first line but does not intersect
        (points(0, 0.5), []),
        # point at shared vertex between 2 lines (but within neither)
        (points(1, 1), []),
        # box not within line
        (box(0, 0, 1, 1), []),
        # buffer intersects 2 lines but not within either
        (buffer(points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects, but both are not within line
        (multipoints([[5, 7], [7, 7]]), []),
        (multipoints([[6.5, 6.5], [7, 7]]), [6]),
    ],
)
def test_query_within_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="within"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (points(0, 0.5), [0]),
        (points(0.5, 0), [0]),
        # midpoint between two polygons intersects both
        (points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (points(1, 1), [1]),
        # box overlaps envelope of 2 polygons but within neither
        (box(0, 0, 1, 1), []),
        # box within polygon
        (box(0, 0, 0.5, 0.5), [0]),
        # larger box intersects 3 polygons but within none
        (box(0, 0, 1.5, 1.5), []),
        # buffer intersects 3 polygons but only within one
        (buffer(points(3, 3), HALF_UNIT_DIAG), [3]),
        # larger buffer overlaps 6 polygons (touches midpoints) but within none
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), []),
        # envelope of points overlaps polygons, but points do not intersect
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint within polygon
        (multipoints([[5, 7], [7, 7]]), []),
        # both points in multipoint within polygon
        (multipoints([[5.25, 5.5], [5.25, 5.0]]), [5]),
    ],
)
def test_query_within_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="within"), expected)


### predicate == 'contains'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (points(0.5, 0.5), []),
        # points intersect
        (points(1, 1), [1]),
        # box contains points (2 are at edges and not contained)
        (box(3, 3, 6, 6), [4, 5]),
        # envelope of buffer contains more points than within buffer
        # due to diagonal distance
        (buffer(points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect
        (multipoints([[5, 5], [7, 7]]), [5, 7]),
        # envelope of points contains points, but points do not intersect
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_contains_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="contains"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any lines (not valid relation)
        (points(0, 0), []),
        # box contains first line (touches edge of 1 but does not contain it)
        (box(0, 0, 1, 1), [0]),
        # buffer intersects 2 lines but contains neither
        (buffer(points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (multipoints([[5, 7], [7, 7]]), []),
        # both points intersect but do not contain any lines (not valid relation)
        (multipoints([[5, 5], [6, 6]]), []),
    ],
)
def test_query_contains_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="contains"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any polygs (not valid relation)
        (points(0, 0), []),
        # box overlaps envelope of 2 polygons but contains neither
        (box(0, 0, 1, 1), []),
        # larger box intersects 3 polygons but contains only one
        (box(0, 0, 2, 2), [1]),
        # buffer overlaps 3 polygons but contains none
        (buffer(points(3, 3), HALF_UNIT_DIAG), []),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_contains_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="contains"), expected)


### predicate == 'overlaps'
# Overlaps only returns results where geometries are of same dimensions
# and do not completely contain each other.
# See: https://postgis.net/docs/ST_Overlaps.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (points(0.5, 0.5), []),
        # points intersect but do not overlap
        (points(1, 1), []),
        # box overlaps points including those at edge but does not overlap
        # (completely contains all points)
        (box(3, 3, 6, 6), []),
        # envelope of buffer contains points, but does not overlap
        (buffer(points(3, 3), 1), []),
        # multipoints intersect but do not overlap (both completely contain each other)
        (multipoints([[5, 5], [7, 7]]), []),
        # envelope of points contains points in tree, but points do not intersect
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects but does not overlap
        # the intersecting point from multipoint completely contains point in tree
        (multipoints([[5, 7], [7, 7]]), []),
    ],
)
def test_query_overlaps_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="overlaps"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects line but is completely contained by it
        (points(0, 0), []),
        # box overlaps second line (contains first line)
        # but of different dimensions so does not overlap
        (box(0, 0, 1.5, 1.5), []),
        # buffer intersects 2 lines but of different dimensions so does not overlap
        (buffer(points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (multipoints([[5, 7], [7, 7]]), []),
        # both points intersect but different dimensions
        (multipoints([[5, 5], [6, 6]]), []),
    ],
)
def test_query_overlaps_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="overlaps"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not overlap any polygons (different dimensions)
        (points(0, 0), []),
        # box overlaps 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box intersects 3 polygons and contains one
        (box(0, 0, 2, 2), [0, 2]),
        # buffer overlaps 3 polygons and contains 1
        (buffer(points(3, 3), HALF_UNIT_DIAG), [2, 4]),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (buffer(points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 4, 5]),
        # one of two points intersects but different dimensions
        (multipoints([[5, 7], [7, 7]]), []),
    ],
)
def test_query_overlaps_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="overlaps"), expected)


### predicate == 'crosses'
# Only valid for certain geometry combinations
# See: https://postgis.net/docs/ST_Crosses.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points intersect but not valid relation
        (points(1, 1), []),
        # all points of result from tree are in common with box
        (box(3, 3, 6, 6), []),
        # all points of result from tree are in common with buffer
        (buffer(points(3, 3), 1), []),
        # only one point of multipoint intersects but not valid relation
        (multipoints([[5, 7], [7, 7]]), []),
    ],
)
def test_query_crosses_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="crosses"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line but is completely in common with line
        (points(0, 0), []),
        # box overlaps envelope of first 2 lines, contains first and crosses second
        (box(0, 0, 1.5, 1.5), [1]),
        # buffer intersects 2 lines
        (buffer(points(3, 3), 0.5), [2, 3]),
        # buffer crosses line
        (buffer(points(2, 1), 1), [1]),
        # envelope of points overlaps lines but intersects none
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (multipoints([[5, 7], [7, 7], [7, 8]]), []),
    ],
)
def test_query_crosses_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="crosses"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon but not valid relation
        (points(0, 0.5), []),
        # box overlaps 2 polygons but not valid relation
        (box(0, 0, 1.5, 1.5), []),
        # buffer overlaps 3 polygons but not valid relation
        (buffer(points(3, 3), HALF_UNIT_DIAG), []),
        # only one point of multipoint within
        (multipoints([[5, 7], [7, 7], [7, 8]]), [7]),
    ],
)
def test_query_crosses_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="crosses"), expected)


### predicate == 'touches'
# See: https://postgis.net/docs/ST_Touches.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (points(0.5, 0.5), []),
        # points intersect but not valid relation
        (points(1, 1), []),
        # box contains points but touches only those at edges
        (box(3, 3, 6, 6), [3, 6]),
        # buffer completely contains point in tree
        (buffer(points(3, 3), 1), []),
        # buffer intersects 2 points but touches only one
        (buffer(points(0, 1), 1), [1]),
        # multipoints intersect but not valid relation
        (multipoints([[5, 5], [7, 7]]), []),
    ],
)
def test_query_touches_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="touches"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (points(0, 0), [0]),
        # point is within line
        (points(0.5, 0.5), []),
        # point at shared vertex between 2 lines
        (points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [1]),
        # buffer intersects 2 lines but does not touch edges of either
        (buffer(points(3, 3), 0.5), []),
        # buffer intersects midpoint of line at tangent but there is a little overlap
        # due to precision issues
        (buffer(points(2, 1), HALF_UNIT_DIAG), []),
        # envelope of points overlaps lines but intersects none
        (multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects at vertex between lines
        (multipoints([[5, 7], [7, 7], [7, 8]]), [6, 7]),
    ],
)
def test_query_touches_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="touches"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (points(0, 0.5), []),
        # point is at edge of first polygon
        (points(HALF_UNIT_DIAG + EPS, 0), [0]),
        # box overlaps envelope of 2 polygons does not touch any at edge
        (box(0, 0, 1, 1), []),
        # box overlaps 2 polygons and touches edge of first
        (box(HALF_UNIT_DIAG + EPS, 0, 2, 2), [0]),
        # buffer overlaps 3 polygons but does not touch any at edge
        (buffer(points(3, 3), HALF_UNIT_DIAG + EPS), []),
        # only one point of multipoint within polygon but does not touch
        (multipoints([[0, 0], [7, 7], [7, 8]]), []),
    ],
)
def test_query_touches_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="touches"), expected)


### Bulk query tests
def test_query_bulk_wrong_dimensions(tree):
    with pytest.raises(TypeError, match="Array should be one dimensional") as ex:
        tree.query_bulk([[points(0.5, 0.5)]])


@pytest.mark.parametrize("geometry", [[], "foo", 1])
def test_query_bulk_wrong_type(tree, geometry):
    with pytest.raises(TypeError, match="Array should be of object dtype") as ex:
        tree.query_bulk(geometry)


def test_query_bulk_empty(tree):
    assert tree.query_bulk([empty]).size == 0


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        ([points(0.5, 0.5)], [[], []]),
        # points intersect
        ([points(1, 1)], [[0], [1]]),
        # first and last points intersect
        ([points(1, 1), points(-1, -1), points(2, 2)], [[0, 2], [1, 2]],),
        # box contains points
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # bigger box contains more points
        ([box(5, 5, 15, 15)], [[0, 0, 0, 0, 0], [5, 6, 7, 8, 9]]),
        # first and last boxes contains points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(5, 5, 15, 15)],
            [[0, 0, 2, 2, 2, 2, 2], [0, 1, 5, 6, 7, 8, 9]],
        ),
        # envelope of buffer contains points
        ([buffer(points(3, 3), 1)], [[0, 0, 0], [2, 3, 4]]),
        # envelope of points contains points
        ([multipoints([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
    ],
)
def test_query_bulk_points(tree, geometry, expected):
    assert_array_equal(tree.query_bulk(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        ([points(0, 0)], [[0], [0]]),
        ([points(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line
        ([points(0, 0.5)], [[0], [0]]),
        # point at shared vertex between 2 lines
        ([points(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # envelope of buffer overlaps envelope of 2 lines
        ([buffer(points(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # envelope of points overlaps 5 lines (touches edge of 2 envelopes)
        ([multipoints([[5, 7], [7, 5]])], [[0, 0, 0, 0], [4, 5, 6, 7]]),
    ],
)
def test_query_bulk_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query_bulk(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects edge of envelopes of 2 polygons
        ([points(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        ([points(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap envelope of 2 polyons
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2], [0, 1, 2, 3]],
        ),
        # larger box overlaps envelope of 3 polygons
        ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        # envelope of buffer overlaps envelope of 3 polygons
        ([buffer(points(3, 3), HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # envelope of larger buffer overlaps envelope of 6 polygons
        (
            [buffer(points(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        ),
        # envelope of points overlaps 3 polygons
        ([multipoints([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
    ],
)
def test_query_bulk_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query_bulk(geometry), expected)


def test_query_invalid_predicate(tree):
    with pytest.raises(ValueError):
        tree.query_bulk(points(1, 1), predicate="bad_predicate")


### predicate == 'intersects'

# TEMPORARY xfail: MultiPoint intersects with prepared geometries does not work
# properly on GEOS 3.5.x; it was fixed in 3.6+
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        ([points(0.5, 0.5)], [[], []]),
        # points intersect
        ([points(1, 1)], [[0], [1]]),
        # box contains points
        ([box(3, 3, 6, 6)], [[0, 0, 0, 0], [3, 4, 5, 6]]),
        # first and last boxes contain points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(3, 3, 6, 6)],
            [[0, 0, 2, 2, 2, 2], [0, 1, 3, 4, 5, 6]],
        ),
        # envelope of buffer contains more points than intersect buffer
        # due to diagonal distance
        ([buffer(points(3, 3), 1)], [[0], [3]]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        ([buffer(points(3, 3), 3 * HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]],),
        # multipoints intersect
        pytest.param(
            [multipoints([[5, 5], [7, 7]])],
            [[0, 0], [5, 7]],
            marks=pytest.mark.xfail(reason="GEOS 3.5"),
        ),
        # envelope of points contains points, but points do not intersect
        ([multipoints([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        pytest.param(
            [multipoints([[5, 7], [7, 7]])],
            [[0], [7]],
            marks=pytest.mark.xfail(reason="GEOS 3.5"),
        ),
    ],
)
def test_query_bulk_intersects_points(tree, geometry, expected):
    assert_array_equal(tree.query_bulk(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        ([points(0, 0)], [[0], [0]]),
        ([points(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line but does not intersect
        ([points(0, 0.5)], [[], []]),
        # point at shared vertex between 2 lines
        ([points(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap multiple lines each
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2, 2], [0, 1, 1, 2, 3]],
        ),
        # buffer intersects 2 lines
        ([buffer(points(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # buffer intersects midpoint of line at tangent
        ([buffer(points(2, 1), HALF_UNIT_DIAG)], [[0], [1]]),
        # envelope of points overlaps lines but intersects none
        ([multipoints([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        ([multipoints([[5, 7], [7, 7]])], [[0, 0], [6, 7]]),
    ],
)
def test_query_bulk_intersects_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query_bulk(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        ([points(0, 0.5)], [[0], [0]]),
        ([points(0.5, 0)], [[0], [0]]),
        # midpoint between two polygons intersects both
        ([points(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        ([points(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2], [0, 1, 2, 3]],
        ),
        # larger box intersects 3 polygons
        ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        # buffer overlaps 3 polygons
        ([buffer(points(3, 3), HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # larger buffer overlaps 6 polygons (touches midpoints)
        (
            [buffer(points(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        ),
        # envelope of points overlaps polygons, but points do not intersect
        ([multipoints([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint within polygon
        ([multipoints([[5, 7], [7, 7]])], [[0], [7]]),
    ],
)
def test_query_bulk_intersects_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query_bulk(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize("geometry", ["I am not a geometry", ["I am not a geometry"]])
def test_nearest_no_geom(tree, geometry):
    with pytest.raises(TypeError):
        tree.nearest(geometry)


@pytest.mark.parametrize("geometry,expected", [(None, [[], []]), ([None], [[], []])])
def test_nearest_none(tree, geometry, expected):
    assert_array_equal(tree.nearest(geometry), expected)


@pytest.mark.parametrize("geometry,expected", [(empty, [[], []]), ([empty], [[], []])])
def test_nearest_empty(tree, geometry, expected):
    assert_array_equal(tree.nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (points(0.25, 0.25), [[0], [0]]),
        (points(0.5, 0.5), [[0], [0]]),
        (points(0.75, 0.75), [[0], [1]]),
        (points(1, 1), [[0], [1]]),
        ([points(1, 1), points(0, 0)], [[0, 1], [1, 0]]),
        ([points(1, 1), points(0.25, 1)], [[0, 1], [1, 1]]),
        ([points(-10, -10), points(100, 100)], [[0, 1], [0, 9]]),
        (box(0, 0, 1, 1), [[0], [0]]),
        (box(0.5, 0.5, 0.75, 0.75), [[0], [1]]),
        (buffer(points(2.5, 2.5), HALF_UNIT_DIAG), [[0], [2]]),
        (buffer(points(3, 3), HALF_UNIT_DIAG), [[0], [3]]),
        (multipoints([[5, 5], [7, 7]]), [[0], [5]]),
        (multipoints([[5.5, 5], [7, 7]]), [[0], [7]]),
        (multipoints([[5, 7], [7, 5]]), [[0], [6]]),
    ],
)
def test_nearest_points(tree, geometry, expected):
    assert_array_equal(tree.nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (points(0.5, 0.5), [[0], [0]]),
        (points(2, 2), [[0], [1]]),
        (box(0, 0, 1, 1), [[0], [0]]),
        (box(0.5, 0.5, 1.5, 1.5), [[0], [0]]),
        ([box(0, 0, 1, 1), box(3, 3, 5, 5)], [[0, 1], [0, 2]]),
        (buffer(points(2.5, 2.5), HALF_UNIT_DIAG), [[0], [1]]),
        (buffer(points(3, 3), HALF_UNIT_DIAG), [[0], [2]]),
        (multipoints([[5, 5], [7, 7]]), [[0], [4]]),
        (multipoints([[5.5, 5], [7, 7]]), [[0], [6]]),
        (multipoints([[5, 7], [7, 5]]), [[0], [5]]),
    ],
)
def test_nearest_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (points(0, 0), [[0], [0]]),
        (points(0.5, 0.5), [[0], [0]]),
        (points(2, 2), [[0], [2]]),
        (box(0, 0, 1, 1), [[0], [0]]),
        (box(0.5, 0.5, 1.5, 1.5), [[0], [0]]),
        ([box(0, 0, 1, 1), box(3, 3, 5, 5)], [[0, 1], [0, 3]]),
        (buffer(points(2.5, 2.5), HALF_UNIT_DIAG), [[0], [2]]),
        (buffer(points(3, 3), HALF_UNIT_DIAG), [[0], [2]]),
        (multipoints([[5, 5], [7, 7]]), [[0], [5]]),
        (multipoints([[5.5, 5], [7, 7]]), [[0], [5]]),
        (multipoints([[5, 7], [7, 5]]), [[0], [6]]),
    ],
)
def test_nearest_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.nearest(geometry), expected)
