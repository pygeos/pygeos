from enum import IntEnum
import numpy as np
from . import lib
from . import Geometry  # NOQA

__all__ = [
    "GeometryType",
    "get_type_id",
    "get_dimensions",
    "get_coordinate_dimensions",
    "get_num_coordinates",
    "get_srid",
    "set_srid",
    "get_x",
    "get_y",
    "get_exterior_ring",
    "get_num_points",
    "get_num_interior_rings",
    "get_num_geometries",
    "get_point",
    "get_interior_ring",
    "get_geometry",
]


class GeometryType(IntEnum):
    """The enumeration of GEOS geometry types"""

    NAG = -1
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


# generic


def get_type_id(geometry):
    """Returns the type ID of a geometry.

    - None is -1
    - POINT is 0
    - LINESTRING is 1
    - LINEARRING is 2
    - POLYGON is 3
    - MULTIPOINT is 4
    - MULTILINESTRING is 5
    - MULTIPOLYGON is 6
    - GEOMETRYCOLLECTION is 7

    Parameters
    ----------
    geometry : Geometry or array_like

    See also
    --------
    GeometryType

    Examples
    --------
    >>> get_type_id(Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)"))
    1
    >>> get_type_id([Geometry("POINT (1 2)"), Geometry("POINT (1 2)")]).tolist()
    [0, 0]
    """
    return lib.get_type_id(geometry)


def get_dimensions(geometry):
    """Returns the inherent dimensionality of a geometry.

    The inherent dimension is 0 for points, 1 for linestrings and linearrings,
    and 2 for polygons. For geometrycollections it is the max of the containing
    elements. Empty and None geometries return -1.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> get_dimensions(Geometry("POINT (0 0)"))
    0
    >>> get_dimensions(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    2
    >>> get_dimensions(Geometry("GEOMETRYCOLLECTION (POINT(0 0), LINESTRING(0 0, 1 1))"))
    1
    >>> get_dimensions(Geometry("GEOMETRYCOLLECTION EMPTY"))
    -1
    >>> get_dimensions(None)
    -1
    """
    return lib.get_dimensions(geometry)


def get_coordinate_dimensions(geometry):
    """Returns the dimensionality of the coordinates in a geometry (2 or 3).

    Returns -1 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> get_coordinate_dimensions(Geometry("POINT (0 0)"))
    2
    >>> get_coordinate_dimensions(Geometry("POINT Z (0 0 0)"))
    3
    >>> get_coordinate_dimensions(None)
    -1
    """
    return lib.get_coordinate_dimensions(geometry)


def get_num_coordinates(geometry):
    """Returns the total number of coordinates in a geometry.

    Returns -1 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> get_num_coordinates(Geometry("POINT (0 0)"))
    1
    >>> get_num_coordinates(Geometry("POINT Z (0 0 0)"))
    1
    >>> get_num_coordinates(Geometry("GEOMETRYCOLLECTION (POINT(0 0), LINESTRING(0 0, 1 1))"))
    3
    >>> get_num_coordinates(None)
    -1
    """
    return lib.get_num_coordinates(geometry)


def get_srid(geometry):
    """Returns the SRID of a geometry.

    Returns -1 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like

    See also
    --------
    set_srid

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> with_srid = set_srid(point, 4326)
    >>> get_srid(point)
    0
    >>> get_srid(with_srid)
    4326
    """
    return lib.get_srid(geometry)


def set_srid(geometry, srid):
    """Returns a geometry with its SRID set.

    Parameters
    ----------
    geometry : Geometry or array_like
    srid : int

    See also
    --------
    get_srid

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> with_srid = set_srid(point, 4326)
    >>> get_srid(point)
    0
    >>> get_srid(with_srid)
    4326
    """
    return lib.set_srid(geometry, np.intc(srid))


# points


def get_x(point):
    """Returns the x-coordinate of a point

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries will result in NaN being returned.

    See also
    --------
    get_y

    Examples
    --------
    >>> get_x(Geometry("POINT (1 2)"))
    1.0
    >>> get_x(Geometry("MULTIPOINT (1 1, 1 2)"))
    nan
    """
    return lib.get_x(point)


def get_y(point):
    """Returns the y-coordinate of a point

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries will result in NaN being returned.

    See also
    --------
    get_x

    Examples
    --------
    >>> get_y(Geometry("POINT (1 2)"))
    2.0
    >>> get_y(Geometry("MULTIPOINT (1 1, 1 2)"))
    nan
    """
    return lib.get_y(point)


# linestrings


def get_point(geometry, index):
    """Returns the nth point of a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the linestring backwards.

    See also
    --------
    get_num_points

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)")
    >>> get_point(line, 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_point(line, -2)
    <pygeos.Geometry POINT (2 2)>
    >>> get_point(line, [0, 3]).tolist()
    [<pygeos.Geometry POINT (0 0)>, <pygeos.Geometry POINT (3 3)>]
    >>> get_point(Geometry("LINEARRING (0 0, 1 1, 2 2, 0 0)"), 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_point(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"), 1) is None
    True
    >>> get_point(Geometry("POINT (1 1)"), 0) is None
    True
    """
    return lib.get_point(geometry, np.intc(index))


def get_num_points(geometry):
    """Returns number of points in a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of points in geometries other than linestring or linearring
        equals zero.

    See also
    --------
    get_point
    get_num_geometries

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)")
    >>> get_num_points(line)
    4
    >>> get_num_points(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"))
    0
    """
    return lib.get_num_points(geometry)


# polygons


def get_exterior_ring(geometry):
    """Returns the exterior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like

    See also
    --------
    get_interior_ring

    Examples
    --------
    >>> get_exterior_ring(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    <pygeos.Geometry LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>
    >>> get_exterior_ring(Geometry("POINT (1 1)")) is None
    True
    """
    return lib.get_exterior_ring(geometry)


def get_interior_ring(geometry, index):
    """Returns the nth interior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the interior rings backwards.

    See also
    --------
    get_exterior_ring
    get_num_interior_rings

    Examples
    --------
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_interior_ring(polygon_with_hole, 0)
    <pygeos.Geometry LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>
    >>> get_interior_ring(Geometry("POINT (1 1)"), 0) is None
    True
    """
    return lib.get_interior_ring(geometry, np.intc(index))


def get_num_interior_rings(geometry):
    """Returns number of internal rings in a polygon

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of interior rings in non-polygons equals zero.

    See also
    --------
    get_exterior_ring
    get_interior_ring

    Examples
    --------
    >>> polygon = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))")
    >>> get_num_interior_rings(polygon)
    0
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_num_interior_rings(polygon_with_hole)
    1
    >>> get_num_interior_rings(Geometry("POINT (1 1)"))
    0
    """
    return lib.get_num_interior_rings(geometry)


# collections


def get_geometry(geometry, index):
    """Returns the nth geometry from a collection of geometries.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the collection backwards.

    Notes
    -----
    - simple geometries act as length-1 collections
    - out-of-range values return None

    See also
    --------
    get_num_geometries

    Examples
    --------
    >>> multipoint = Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)")
    >>> get_geometry(multipoint, 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_geometry(multipoint, -1)
    <pygeos.Geometry POINT (3 3)>
    >>> get_geometry(multipoint, 5) is None
    True
    >>> get_geometry(Geometry("POINT (1 1)"), 0)
    <pygeos.Geometry POINT (1 1)>
    >>> get_geometry(Geometry("POINT (1 1)"), 1) is None
    True
    """
    return lib.get_geometry(geometry, np.intc(index))


def get_num_geometries(geometry):
    """Returns number of geometries in a collection.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of geometries in points, linestrings, linearrings and
        polygons equals one.

    See also
    --------
    get_num_points
    get_geometry

    Examples
    --------
    >>> get_num_geometries(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"))
    4
    >>> get_num_geometries(Geometry("POINT (1 1)"))
    1
    """
    return lib.get_num_geometries(geometry)
