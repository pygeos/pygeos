import numpy as np
from . import lib, Geometry, GeometryType, box
from .decorators import requires_geos, UnsupportedGEOSOperation
from .decorators import multithreading_enabled

__all__ = [
    "difference",
    "intersection",
    "intersection_all",
    "symmetric_difference",
    "symmetric_difference_all",
    "union",
    "union_all",
    "coverage_union",
    "coverage_union_all",
]

@multithreading_enabled
def difference(a, b, grid_size=None, **kwargs):
    """Returns the part of geometry A that does not intersect with geometry B.

    If grid_size is nonzero, input coordinates will be snapped to a precision grid of that
    size and resulting coordinates will be snapped to that same grid.  If 0, this
    operation will use double precision coordinates.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional (default: None).
        precision grid size; requires GEOS >= 3.9.0.

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 2 2)")
    >>> difference(line, Geometry("LINESTRING (1 1, 3 3)"))
    <pygeos.Geometry LINESTRING (0 0, 1 1)>
    >>> difference(line, Geometry("LINESTRING EMPTY"))
    <pygeos.Geometry LINESTRING (0 0, 2 2)>
    >>> difference(line, None) is None
    True
    >>> difference(box(0,0,2,2),box(1,1,3,3))
    <pygeos.Geometry POLYGON ((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0))>
    >>> difference(box(0.1,0.2,2.1,2.1),box(1,1,3,3), grid_size=1) # doctest: +SKIP
    <pygeos.Geometry POLYGON ((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0))>
    """

    if grid_size is not None:
        if lib.geos_version < (3,9,0):
            raise UnsupportedGEOSOperation("grid_size parameter requires GEOS >= 3.9.0")

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.difference_prec(a, b, grid_size, **kwargs)

    return lib.difference(a, b, **kwargs)

@multithreading_enabled
def intersection(a, b, grid_size=None, **kwargs):
    """Returns the geometry that is shared between input geometries.

    If grid_size is nonzero, input coordinates will be snapped to a precision grid of that
    size and resulting coordinates will be snapped to that same grid.  If 0, this
    operation will use double precision coordinates.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional (default: None).
        precision grid size; requires GEOS >= 3.9.0.

    See also
    --------
    intersection_all

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 2 2)")
    >>> intersection(line, Geometry("LINESTRING(1 1, 3 3)"))
    <pygeos.Geometry LINESTRING (1 1, 2 2)>
    >>> intersection(box(0,0,2,2),box(1,1,3,3))
    <pygeos.Geometry POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))>
    >>> intersection(box(0.1,0.2,2.1,2.1),box(1,1,3,3), grid_size=1) # doctest: +SKIP
    <pygeos.Geometry POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))>
    """

    if grid_size is not None:
        if lib.geos_version < (3,9,0):
            raise UnsupportedGEOSOperation("grid_size parameter requires GEOS >= 3.9.0")

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.intersection_prec(a, b, grid_size, **kwargs)

    return lib.intersection(a, b, **kwargs)

@multithreading_enabled
def intersection_all(geometries, axis=0, **kwargs):
    """Returns the intersection of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, None is returned.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    intersection

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> intersection_all([line_1, line_2])
    <pygeos.Geometry LINESTRING (1 1, 2 2)>
    >>> intersection_all([[line_1, line_2, None]], axis=1).tolist()
    [<pygeos.Geometry LINESTRING (1 1, 2 2)>]
    """
    return lib.intersection.reduce(geometries, axis=axis, **kwargs)

@multithreading_enabled
def symmetric_difference(a, b, grid_size=None, **kwargs):
    """Returns the geometry that represents the portions of input geometries
    that do not intersect.

    If grid_size is nonzero, input coordinates will be snapped to a precision grid of that
    size and resulting coordinates will be snapped to that same grid.  If 0, this
    operation will use double precision coordinates.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional (default: None).
        precision grid size; requires GEOS >= 3.9.0.

    See also
    --------
    symmetric_difference_all

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 2 2)")
    >>> symmetric_difference(line, Geometry("LINESTRING(1 1, 3 3)"))
    <pygeos.Geometry MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> symmetric_difference(box(0,0,2,2),box(1,1,3,3))
    <pygeos.Geometry MULTIPOLYGON (((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0)), ((1 3,...>
    >>> symmetric_difference(box(0.1,0.2,2.1,2.1),box(1,1,3,3), grid_size=1) # doctest: +SKIP
    <pygeos.Geometry MULTIPOLYGON (((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0)), ((1 3,...>
    """

    if grid_size is not None:
        if lib.geos_version < (3,9,0):
            raise UnsupportedGEOSOperation("grid_size parameter requires GEOS >= 3.9.0")

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.symmetric_difference_prec(a, b, grid_size, **kwargs)

    return lib.symmetric_difference(a, b, **kwargs)

@multithreading_enabled
def symmetric_difference_all(geometries, axis=0, **kwargs):
    """Returns the symmetric difference of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, None is returned.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    symmetric_difference

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> symmetric_difference_all([line_1, line_2])
    <pygeos.Geometry MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> symmetric_difference_all([[line_1, line_2, None]], axis=1).tolist()
    [<pygeos.Geometry MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>]
    """
    return lib.symmetric_difference.reduce(geometries, axis=axis, **kwargs)

@multithreading_enabled
def union(a, b, grid_size=None, **kwargs):
    """Merges geometries into one.

    If grid_size is nonzero, input coordinates will be snapped to a precision grid of that
    size and resulting coordinates will be snapped to that same grid.  If 0, this
    operation will use double precision coordinates.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    grid_size : float, optional (default: None).
        precision grid size; requires GEOS >= 3.9.0.

    See also
    --------
    union_all

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 2 2)")
    >>> union(line, Geometry("LINESTRING(2 2, 3 3)"))
    <pygeos.Geometry MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> union(line, None) is None
    True
    >>> union(box(0,0,2,2),box(1,1,3,3))
    <pygeos.Geometry POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    >>> union(box(0.1,0.2,2.1,2.1),box(1,1,3,3), grid_size=1) # doctest: +SKIP
    <pygeos.Geometry POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    """

    if grid_size is not None:
        if lib.geos_version < (3,9,0):
            raise UnsupportedGEOSOperation("grid_size parameter requires GEOS >= 3.9.0")

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.union_prec(a, b, grid_size, **kwargs)

    return lib.union(a, b, **kwargs)

@multithreading_enabled
def union_all(geometries, grid_size=None, axis=0, **kwargs):
    """Returns the union of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, None is returned.

    If grid_size is nonzero, input coordinates will be snapped to a precision grid of that
    size and resulting coordinates will be snapped to that same grid.  If 0, this
    operation will use double precision coordinates.

    Parameters
    ----------
    geometries : array_like
    grid_size : float, optional (default: None).
        precision grid size; requires GEOS >= 3.9.0.
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    union

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(2 2, 3 3)")
    >>> union_all([line_1, line_2])
    <pygeos.Geometry MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> union_all([[line_1, line_2, None]], axis=1).tolist()
    [<pygeos.Geometry MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>]
    >>> union_all([box(0,0,2,2),box(1,1,3,3)])
    <pygeos.Geometry POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    >>> union_all([box(0.1,0.2,2.1,2.1),box(1,1,3,3)], grid_size=1) # doctest: +SKIP
    <pygeos.Geometry POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>

    """
    # for union_all, GEOS provides an efficient route through first creating
    # GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)

    if grid_size is not None:
        if lib.geos_version < (3,9,0):
            raise UnsupportedGEOSOperation("grid_size parameter requires GEOS >= 3.9.0")

        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        result = lib.unary_union_prec(collections, grid_size, **kwargs)

    else:
        result = lib.unary_union(collections, **kwargs)
    # for consistency with other _all functions, we replace GEOMETRY COLLECTION EMPTY
    # if the original collection had no geometries
    only_none = lib.get_num_geometries(collections) == 0
    if np.isscalar(only_none):
        return result if not only_none else None
    else:
        result[only_none] = None
        return result


@requires_geos("3.8.0")
@multithreading_enabled
def coverage_union(a, b, **kwargs):
    """Merges multiple polygons into one. This is an optimized version of
    union which assumes the polygons to be non-overlapping.

    Requires at least GEOS 3.8.0.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    See also
    --------
    coverage_union_all

    Examples
    --------
    >>> from pygeos.constructive import normalize
    >>> polygon = Geometry("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    >>> normalize(coverage_union(polygon, Geometry("POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))")))
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>

    Union with None returns same polygon
    >>> normalize(coverage_union(polygon, None))
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    return coverage_union_all([a, b], **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def coverage_union_all(geometries, axis=0, **kwargs):
    """Returns the union of multiple polygons of a geometry collection.
    This is an optimized version of union which assumes the polygons
    to be non-overlapping.

    Requires at least GEOS 3.8.0.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    coverage_union

    Examples
    --------
    >>> from pygeos.constructive import normalize
    >>> polygon_1 = Geometry("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    >>> polygon_2 = Geometry("POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))")
    >>> normalize(coverage_union_all([polygon_1, polygon_2]))
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>
    """
    # coverage union in GEOS works over GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)
    return lib.coverage_union(collections, **kwargs)
