# distutils: define_macros=GEOS_USE_ONLY_R_API

from libc.stdio cimport *
from libc.float cimport DBL_MAX, DBL_MIN
from libc.stdint cimport uint8_t, uint64_t
from cpython cimport PyObject

cimport cython
cimport numpy as np

from pygeos._geometry cimport _bounds
from pygeos._geos cimport (
    GEOSCoordSequence,
    GEOSCoordSeq_create_r,
    GEOSCoordSeq_destroy_r,
    GEOSContextHandle_t,
    GEOSGeometry,
    GEOSGeomTypeId_r,
    GEOSGeom_clone_r,
    GEOSGetNumCoordinates_r,
    GEOSGetGeometryN_r,
    GEOSGetNumGeometries_r,
    GEOSisEmpty_r,
    GEOSGeom_destroy_r,
    get_geos_handle,
    GEOSCoordSeq_setXY_r,
    GEOSGeom_createLinearRing_r,
    GEOSGeom_createPolygon_r,
    GEOSIntersection_r,
    GEOSSimplify_r,
    GEOSGeom_getDimensions_r
)

from pygeos._pygeos_api cimport (
    import_pygeos_c_api,
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry
)
from pygeos._vector cimport (
    GeometryVector,
    IndexVector
)

# initialize PyGEOS C API
import_pygeos_c_api()

cdef uint8_t MAX_RECURSION_DEPTH = 50


# requires GEOS >= 3.8
cdef GEOSGeometry* _box(
    GEOSContextHandle_t geos_handle,
    double xmin,
    double ymin,
    double xmax,
    double ymax
) nogil except NULL:
    """Creates a polygon starting at bottom left oriented counterclockwise
    (does not require gil).

    Requires GEOS >= 3.8.

    Parameters
    ----------
    geos_handle : GEOSContextHandle_t
    geom : GEOSGeometry pointer
    xmin : double
    ymin : double
    xmax : double
    ymax : double

    Returns
    -------
    GEOSGeometry pointer
        Caller becomes owner of this geometry.  NULL is returned if create operation fails.

    See also
    --------
    pygeos.box : ufunc that creates polygons from 4 coordinates
    """
    cdef GEOSCoordSequence *coords = NULL
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *ring = NULL

    # Construct coordinate sequence and set vertices
    coords = GEOSCoordSeq_create_r(geos_handle, 5, 2)
    if coords == NULL:
        return NULL

    if not (
        GEOSCoordSeq_setXY_r(geos_handle, coords, 0, xmin, ymin)
        and GEOSCoordSeq_setXY_r(geos_handle, coords, 1, xmax, ymin)
        and GEOSCoordSeq_setXY_r(geos_handle, coords, 2, xmax, ymax)
        and GEOSCoordSeq_setXY_r(geos_handle, coords, 3, xmin, ymax)
        and GEOSCoordSeq_setXY_r(geos_handle, coords, 4, xmin, ymin)
    ):
        if coords != NULL:
            GEOSCoordSeq_destroy_r(geos_handle, coords)

        return NULL

    # Construct linear ring then use to construct polygon
    # NOTE: coords are owned by ring
    ring = GEOSGeom_createLinearRing_r(geos_handle, coords)
    if ring == NULL:
        return NULL

    # NOTE: ring is owned by polygon
    geom = GEOSGeom_createPolygon_r(geos_handle, ring, NULL, 0)
    if geom == NULL:
        if ring != NULL:
            GEOSGeom_destroy_r(geos_handle, ring)

        return NULL

    return geom


# returns count of geometries
# max_vertices maybe uint32, and check bounds on input below
cdef Py_ssize_t _subdivide_geometry(
    GEOSContextHandle_t geos_handle,
    const GEOSGeometry *geom,
    int geom_dimension,
    int max_vertices,
    uint8_t depth,
    void *out_geom_vec
) nogil:
    """Recursively subdivides geometry.

    This will stop subdividing geometry when the following conditions occur:
    * geometry is NULL (discards geometry, returns 0)
    * dimension of geometry is less than parent (discards geometry, returns 0)
    * recursion limit is reached (saves geometry, returns 1)
    * geometry is a point or empty (saves geometry, returns 1)
    * geometry has fewer coordinates than max_vertices (saves geometry, returns 1)

    Will recurse into GeometryCollection and multi-part geometries.

    Will subdivide the geometry into 4 tiles based on the midpoint X and Y values and
    recurse over each tile.

    Parameters
    ----------
    geos_handle : GEOSContextHandle_t
    geom : GEOSGeometry pointer
    geom_dimension : int
        Geometry dimension of parent geometry, to detect dimension collapse.
        point=0, line=1, polygon=2
    max_vertices : int
        Maximum number of vertices to target for subdivided geometry.

    Returns
    -------
    Py_ssize_t
        Count of geometries added by recursion.  Corresponds to count of geometries pushed
        onto out_geom_vec.
    """
    cdef int type_id
    cdef uint8_t stop = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t count = 0
    cdef int num_parts = 0
    cdef int num_coords = 0
    cdef double xmax = DBL_MIN
    cdef double ymax = DBL_MIN
    cdef double xmin = DBL_MAX
    cdef double ymin = DBL_MAX
    cdef double width = 0
    cdef double height = 0
    cdef Py_ssize_t quadrant = 0
    cdef double center_x = DBL_MAX
    cdef double center_y = DBL_MAX
    cdef GEOSGeometry *clip_geom = NULL
    cdef const GEOSGeometry *part = NULL
    cdef const GEOSGeometry *clipped_geom = NULL

    if geom == NULL:
        # All NULLs are filtered before recursion; any within recursion should be ignored.
        return 0

    if GEOSGeom_getDimensions_r(geos_handle, geom) < geom_dimension:
        # If dimension of geometry is less than dimension of parent geometry, it was
        # collapsed by preceding intersection operation; discard it.
        return 0

    type_id = GEOSGeomTypeId_r(geos_handle, geom)

    # Stop recursion when:
    # * recursion limit is reached
    # * geometry is empty or a point type (not subdividable)
    # * number of coordinates is below limit
    if (
        depth > MAX_RECURSION_DEPTH
        or GEOSisEmpty_r(geos_handle, geom)
        or type_id == 0
        or GEOSGetNumCoordinates_r(geos_handle, geom) <= max_vertices
    ):
        out_geom = GEOSGeom_clone_r(geos_handle, geom)
        (<GeometryVector>out_geom_vec).push(out_geom)
        return 1

    depth += 1

    # Multi* or GeometryCollection: recurse over parts
    if type_id >= 4:
        num_parts = GEOSGetNumGeometries_r(geos_handle, geom)

        for part_idx in range(num_parts):
            part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)
            if part != NULL:
                if depth == 0:
                    # calculate dimension to handle mixed GeometryCollection parts
                    # only for top-level Multi* and GeometryCollections
                    geom_dimension = GEOSGeom_getDimensions_r(geos_handle, part)

                count += _subdivide_geometry(
                            geos_handle, part, geom_dimension, max_vertices, depth,
                            out_geom_vec)
        return count


    if _bounds(geos_handle, geom, &xmin, &ymin, &xmax, &ymax) == 0:
        # could not determine bounds of a valid geometry
        # with gil:
        raise RuntimeError("Could not calculate bounds of geometry")

    width = xmax - xmin
    height = ymax - ymin

    if width == 0 or height == 0:
        # Dimension collapse and non-boundable geometries (empty, point) were handled above
        if (width + height) > 0:
            # This handles the case where the geometry is strictly vertical or horizontal;
            # regardless of number of vertices, no point in subdividing further
            out_geom = GEOSGeom_clone_r(geos_handle, geom)
            (<GeometryVector>out_geom_vec).push(out_geom)
            return 1

        # Dimension collapse not caught previously, ignore this geometry
        return 0

    # split into 4 tiles from center of bounds
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    for quadrant in range(4):
        if quadrant == 0:  # bottom left
            clip_geom = _box(geos_handle, xmin, ymin, center_x, center_y)

        elif quadrant == 1:  # top left
            clip_geom = _box(geos_handle, xmin, center_y, center_x, ymax)

        elif quadrant == 2:  # top right
            clip_geom = _box(geos_handle, center_x, center_y, xmax, ymax)

        else:  # bottom right
            clip_geom = _box(geos_handle, center_x, ymin, xmax, center_y)

        if clip_geom == NULL:
            # could not construct clip geom from valid coordinates
            with gil:
                raise RuntimeError("Could not construct clip geometry from coordinates")

        # clip by clip_geom
        # TODO: use precision version (what GEOS version?)
        clipped_geom = GEOSIntersection_r(geos_handle, geom, clip_geom)
        if clipped_geom == NULL:
            if clip_geom != NULL:
                GEOSGeom_destroy_r(geos_handle, clip_geom)

            with gil:
                # TODO: we want the underlying GEOS error message here
                raise RuntimeError("Intersection of geometry and clip geometry failed")

        # simplify geometry
        clipped_geom = GEOSSimplify_r(geos_handle, clipped_geom, 0)
        if clipped_geom == NULL:
            if clip_geom != NULL:
                GEOSGeom_destroy_r(geos_handle, clip_geom)

            with gil:
                # TODO: we want the underlying GEOS error message here
                raise RuntimeError("Simplification of clip geometry failed")

        if GEOSisEmpty_r(geos_handle, clipped_geom) != 0:
            # This shouldn't happen but ignore it in case it does
            continue

        # Recurse into clipped geometry
        # NOTE: this clones clipped_geom as needed, so we need to cleanup ourselves
        count += _subdivide_geometry(
                    geos_handle, clipped_geom, geom_dimension, max_vertices, depth,
                    out_geom_vec)


        GEOSGeom_destroy_r(geos_handle, clip_geom)
        clip_geom = NULL

        GEOSGeom_destroy_r(geos_handle, clipped_geom)
        clipped_geom = NULL


    return count


# TODO: decision: filter to only geometries (filter out empty / NULL)?
# TODO: crosscheck ST_subdivide results
@cython.boundscheck(False)
@cython.wraparound(False)
def subdivide(array, int max_vertices=256):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t i = 0
    cdef int geom_dimension = 0;
    cdef GEOSGeometry *geom = NULL
    cdef Py_ssize_t initial_size = len(array)

    if max_vertices <= 5:
        raise ValueError("max_vertices must be greater than 5")

    with get_geos_handle() as geos_handle, \
         GeometryVector(initial_size) as out_geom_vec, \
         IndexVector(initial_size) as out_idx_vec:

        for geom_idx in range(initial_size):
            if PyGEOS_GetGEOSGeometry(<PyObject *>array[geom_idx], &geom) == 0:
                raise TypeError("One of the arguments is of incorrect type. "
                                "Please provide only Geometry objects.")

            if geom == NULL:
                # input was None
                out_geom_vec.push(NULL)
                out_idx_vec.push(geom_idx)
                continue

            geom_dimension = GEOSGeom_getDimensions_r(geos_handle, geom)

            count = _subdivide_geometry(
                        geos_handle,
                        <const GEOSGeometry*>geom,
                        geom_dimension,
                        max_vertices,
                        0,
                        <void*>out_geom_vec
                    )

            for i in range(count):
                out_idx_vec.push(geom_idx)

        out_idx = out_idx_vec.to_array()
        out_geom = out_geom_vec.to_array(geos_handle)

        return out_geom, out_idx

