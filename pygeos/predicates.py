import warnings

from . import lib
from . import Geometry  # NOQA

__all__ = [
    "has_z",
    "is_closed",
    "is_empty",
    "is_geometry",
    "is_missing",
    "is_ring",
    "is_simple",
    "is_valid",
    "is_valid_input",
    "is_valid_reason",
    "crosses",
    "contains",
    "covered_by",
    "covers",
    "disjoint",
    "equals",
    "intersects",
    "overlaps",
    "touches",
    "within",
    "equals_exact",
]


def has_z(geometry, **kwargs):
    """Returns True if a geometry has a Z coordinate.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> has_z(Geometry("POINT (0 0)"))
    False
    >>> has_z(Geometry("POINT Z (0 0 0)"))
    True
    """
    return lib.has_z(geometry, **kwargs)


def is_closed(geometry, **kwargs):
    """Returns True if a linestring's first and last points are equal.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.

    See also
    --------
    is_ring : Checks additionally if the geometry is simple.

    Examples
    --------
    >>> is_closed(Geometry("LINESTRING (0 0, 1 1)"))
    False
    >>> is_closed(Geometry("LINESTRING(0 0, 0 1, 1 1, 0 0)"))
    True
    >>> is_closed(Geometry("POINT (0 0)"))
    False
    """
    return lib.is_closed(geometry, **kwargs)


def is_empty(geometry, **kwargs):
    """Returns True if a geometry is an empty point, polygon, etc.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    See also
    --------
    is_missing : checks if the object is a geometry

    Examples
    --------
    >>> is_empty(Geometry("POINT EMPTY"))
    True
    >>> is_empty(Geometry("POINT (0 0)"))
    False
    >>> is_empty(None)
    False
    """
    return lib.is_empty(geometry, **kwargs)


def is_geometry(geometry, **kwargs):
    """Returns True if the object is a geometry

    Parameters
    ----------
    geometry : any object or array_like

    See also
    --------
    is_missing : check if an object is missing (None)
    is_valid_input : check if an object is a geometry or None

    Examples
    --------
    >>> is_geometry(Geometry("POINT (0 0)"))
    True
    >>> is_geometry(Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> is_geometry(None)
    False
    >>> is_geometry("text")
    False
    """
    return lib.is_geometry(geometry, **kwargs)


def is_missing(geometry, **kwargs):
    """Returns True if the object is not a geometry (None)

    Parameters
    ----------
    geometry : any object or array_like

    See also
    --------
    is_geometry : check if an object is a geometry
    is_valid_input : check if an object is a geometry or None
    is_empty : checks if the object is an empty geometry

    Examples
    --------
    >>> is_missing(Geometry("POINT (0 0)"))
    False
    >>> is_missing(Geometry("GEOMETRYCOLLECTION EMPTY"))
    False
    >>> is_missing(None)
    True
    >>> is_missing("text")
    False
    """
    return lib.is_missing(geometry, **kwargs)


def is_valid_input(geometry, **kwargs):
    """Returns True if the object is a geometry or None

    Parameters
    ----------
    geometry : any object or array_like

    See also
    --------
    is_geometry : checks if an object is a geometry
    is_missing : checks if an object is None

    Examples
    --------
    >>> is_valid_input(Geometry("POINT (0 0)"))
    True
    >>> is_valid_input(Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> is_valid_input(None)
    True
    >>> is_valid_input(1.0)
    False
    >>> is_valid_input("text")
    False
    """
    return lib.is_valid_input(geometry, **kwargs)


def is_ring(geometry, **kwargs):
    """Returns True if a linestring is closed and simple.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.

    See also
    --------
    is_closed : Checks only if the geometry is closed.
    is_simple : Checks only if the geometry is simple.

    Examples
    --------
    >>> is_ring(Geometry("POINT (0 0)"))
    False
    >>> geom = Geometry("LINESTRING(0 0, 1 1)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (False, True, False)
    >>> geom = Geometry("LINESTRING(0 0, 0 1, 1 1, 0 0)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, True, True)
    >>> geom = Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, False, False)
    """
    return lib.is_ring(geometry, **kwargs)


def is_simple(geometry, **kwargs):
    """Returns True if a Geometry has no anomalous geometric points, such as
    self-intersections or self tangency.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for geometrycollections.

    See also
    --------
    is_ring : Checks additionally if the geometry is closed.

    Examples
    --------
    >>> is_simple(Geometry("POLYGON((1 1, 2 1, 2 2, 1 1))"))
    True
    >>> is_simple(Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)"))
    False
    >>> is_simple(None)
    False
    """
    return lib.is_simple(geometry, **kwargs)


def is_valid(geometry, **kwargs):
    """Returns True if a geometry is well formed.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns False for missing values.

    See also
    --------
    is_valid_reason : Returns the reason in case of invalid.

    Examples
    --------
    >>> is_valid(Geometry("LINESTRING(0 0, 1 1)"))
    True
    >>> is_valid(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    False
    >>> is_valid(Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> is_valid(None)
    False
    """
    # GEOS is valid will emit warnings for invalid geometries. Suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = lib.is_valid(geometry, **kwargs)
    return result


def is_valid_reason(geometry, **kwargs):
    """Returns a string stating if a geometry is valid and if not, why.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns None for missing values.

    See also
    --------
    is_valid : returns True or False

    Examples
    --------
    >>> is_valid_reason(Geometry("LINESTRING(0 0, 1 1)"))
    'Valid Geometry'
    >>> is_valid_reason(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    'Self-intersection[0 0]'
    >>> is_valid_reason(None) is None
    True
    """
    return lib.is_valid_reason(geometry, **kwargs)


def crosses(a, b, **kwargs):
    """Returns True if the intersection of two geometries spatially crosses.

    That is: the geometries have some, but not all interior points in common.
    The geometries must intersect and the intersection must have a
    dimensionality less than the maximum dimension of the two input geometries.
    Additionally, the intersection of the two geometries must not equal either
    of the source geometries.

    Parameters
    ----------
    a, b : Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> crosses(line, Geometry("POINT (0.5 0.5)"))
    False
    >>> crosses(line, Geometry("MULTIPOINT ((0 1), (0.5 0.5))"))
    True
    >>> crosses(line, Geometry("LINESTRING(0 1, 1 0)"))
    True
    >>> crosses(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> crosses(area, line)
    False
    >>> crosses(area, Geometry("LINESTRING(0 0, 2 2)"))
    True
    >>> crosses(area, Geometry("POINT (0.5 0.5)"))
    False
    >>> crosses(area, Geometry("MULTIPOINT ((2 2), (0.5 0.5))"))
    True
    """
    return lib.crosses(a, b, **kwargs)


def contains(a, b, **kwargs):
    """Returns True if geometry B is completely inside geometry A.

    A contains B if no points of B lie in the exterior of A and at least one
    point of the interior of B lies in the interior of A.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    within : ``contains(A, B) == within(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> contains(line, Geometry("POINT (0 0)"))
    False
    >>> contains(line, Geometry("POINT (0.5 0.5)"))
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> contains(area, Geometry("POINT (0 0)"))
    False
    >>> contains(area, line)
    True
    >>> contains(area, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")
    >>> contains(polygon_with_hole, Geometry("POINT(1 1)"))
    True
    >>> contains(polygon_with_hole, Geometry("POINT(2 2)"))
    False
    >>> contains(polygon_with_hole, Geometry("LINESTRING(1 1, 5 5)"))
    False
    >>> contains(area, area)
    True
    >>> contains(area, None)
    False
    """
    return lib.contains(a, b, **kwargs)


def covered_by(a, b, **kwargs):
    """Returns True if no point in geometry A is outside geometry B.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    covers : ``covered_by(A, B) == covers(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> covered_by(Geometry("POINT (0 0)"), line)
    True
    >>> covered_by(Geometry("POINT (0.5 0.5)"), line)
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> covered_by(Geometry("POINT (0 0)"), area)
    True
    >>> covered_by(line, area)
    True
    >>> covered_by(Geometry("LINESTRING(0 0, 2 2)"), area)
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> covered_by(Geometry("POINT(1 1)"), polygon_with_hole)
    True
    >>> covered_by(Geometry("POINT(2 2)"), polygon_with_hole)
    True
    >>> covered_by(Geometry("LINESTRING(1 1, 5 5)"), polygon_with_hole)
    False
    >>> covered_by(area, area)
    True
    >>> covered_by(None, area)
    False
    """
    return lib.covered_by(a, b, **kwargs)


def covers(a, b, **kwargs):
    """Returns True if no point in geometry B is outside geometry A.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    covered_by : ``covers(A, B) == covered_by(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> covers(line, Geometry("POINT (0 0)"))
    True
    >>> covers(line, Geometry("POINT (0.5 0.5)"))
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> covers(area, Geometry("POINT (0 0)"))
    True
    >>> covers(area, line)
    True
    >>> covers(area, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> covers(polygon_with_hole, Geometry("POINT(1 1)"))
    True
    >>> covers(polygon_with_hole, Geometry("POINT(2 2)"))
    True
    >>> covers(polygon_with_hole, Geometry("LINESTRING(1 1, 5 5)"))
    False
    >>> covers(area, area)
    True
    >>> covers(area, None)
    False
    """
    return lib.covers(a, b, **kwargs)


def disjoint(a, b, **kwargs):
    """Returns True if A and B do not share any point in space.

    Disjoint implies that overlaps, touches, within, and intersects are False.
    Note missing (None) values are never disjoint.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    intersects : ``disjoint(A, B) == ~intersects(A, B)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> disjoint(line, Geometry("POINT (0 0)"))
    False
    >>> disjoint(line, Geometry("POINT (0 1)"))
    True
    >>> disjoint(line, Geometry("LINESTRING(0 2, 2 0)"))
    False
    >>> empty = Geometry("GEOMETRYCOLLECTION EMPTY")
    >>> disjoint(line, empty)
    True
    >>> disjoint(empty, empty)
    True
    >>> disjoint(empty, None)
    False
    >>> disjoint(None, None)
    False
    """
    return lib.disjoint(a, b, **kwargs)


def equals(a, b, **kwargs):
    """Returns True if A and B are spatially equal.

    If A is within B and B is within A, A and B are considered equal. The
    ordering of points can be different.

    Parameters
    ----------
    a, b : Geometry or array_like

    See Also
    --------
    equals_exact : Check if A and B are structurally equal given a specified
        tolerance.

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 5 5, 10 10)")
    >>> equals(line, Geometry("LINESTRING(0 0, 10 10)"))
    True
    >>> equals(Geometry("POLYGON EMPTY"), Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> equals(None, None)
    False
    """
    return lib.equals(a, b, **kwargs)


def intersects(a, b, **kwargs):
    """Returns True if A and B share any portion of space.

    Intersects implies that overlaps, touches and within are True.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    disjoint : ``intersects(A, B) == ~disjoint(A, B)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> intersects(line, Geometry("POINT (0 0)"))
    True
    >>> intersects(line, Geometry("POINT (0 1)"))
    False
    >>> intersects(line, Geometry("LINESTRING(0 2, 2 0)"))
    True
    >>> intersects(None, None)
    False
    """
    return lib.intersects(a, b, **kwargs)


def overlaps(a, b, **kwargs):
    """Returns True if A and B intersect, but one does not completely contain
    the other.

    Parameters
    ----------
    a, b : Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> overlaps(line, line)
    False
    >>> overlaps(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> overlaps(line, Geometry("LINESTRING(0.5 0.5, 2 2)"))
    True
    >>> overlaps(line, Geometry("POINT (0.5 0.5)"))
    False
    >>> overlaps(None, None)
    False
    """
    return lib.overlaps(a, b, **kwargs)


def touches(a, b, **kwargs):
    """Returns True if the only points shared between A and B are on the
    boundary of A and B.

    Parameters
    ----------
    a, b : Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 2, 2 0)")
    >>> touches(line, Geometry("POINT(0 2)"))
    True
    >>> touches(line, Geometry("POINT(1 1)"))
    False
    >>> touches(line, Geometry("LINESTRING(0 0, 1 1)"))
    True
    >>> touches(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> touches(area, Geometry("POINT(0.5 0)"))
    True
    >>> touches(area, Geometry("POINT(0.5 0.5)"))
    False
    >>> touches(area, line)
    True
    >>> touches(area, Geometry("POLYGON((0 1, 1 1, 1 2, 0 2, 0 1))"))
    True
    """
    return lib.touches(a, b, **kwargs)


def within(a, b, **kwargs):
    """Returns True if geometry A is completely inside geometry B.

    A is within B if no points of A lie in the exterior of B and at least one
    point of the interior of A lies in the interior of B.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    contains : ``within(A, B) == contains(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> within(Geometry("POINT (0 0)"), line)
    False
    >>> within(Geometry("POINT (0.5 0.5)"), line)
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> within(Geometry("POINT (0 0)"), area)
    False
    >>> within(line, area)
    True
    >>> within(Geometry("LINESTRING(0 0, 2 2)"), area)
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> within(Geometry("POINT(1 1)"), polygon_with_hole)
    True
    >>> within(Geometry("POINT(2 2)"), polygon_with_hole)
    False
    >>> within(Geometry("LINESTRING(1 1, 5 5)"), polygon_with_hole)
    False
    >>> within(area, area)
    True
    >>> within(None, area)
    False
    """
    return lib.within(a, b, **kwargs)


def equals_exact(a, b, tolerance=0.0, **kwargs):
    """Returns True if A and B are structurally equal.

    This method uses exact coordinate equality, which requires coordinates
    to be equal (within specified tolerance) and and in the same order for all
    components of a geometry. This is in contrast with the `equals` function
    which uses spatial (topological) equality.

    Parameters
    ----------
    a, b : Geometry or array_like
    tolerance : float or array_like

    See Also
    --------
    equals : Check if A and B are spatially equal.

    Examples
    --------
    >>> point1 = Geometry("POINT(50 50)")
    >>> point2 = Geometry("POINT(50.1 50.1)")
    >>> equals_exact(point1, point2)
    False
    >>> equals_exact(point1, point2, tolerance=0.2)
    True
    >>> equals_exact(point1, None, tolerance=0.2)
    False

    Difference between structucal and spatial equality:

    >>> polygon1 = Geometry("POLYGON((0 0, 1 1, 0 1, 0 0))")
    >>> polygon2 = Geometry("POLYGON((0 0, 0 1, 1 1, 0 0))")
    >>> equals_exact(polygon1, polygon2)
    False
    >>> equals(polygon1, polygon2)
    True
    """
    return lib.equals_exact(a, b, tolerance, **kwargs)
