# distutils: define_macros=GEOS_USE_ONLY_R_API


from libc.stdio cimport *
from libc.float cimport DBL_MAX, DBL_MIN
from libc.stdint cimport uint8_t, uint64_t
from cpython cimport PyObject
cimport cython

import numpy as np
cimport numpy as np
import pygeos

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
    GEOSGeom_getXMax_r,
    GEOSGeom_getYMax_r,
    GEOSGeom_getXMin_r,
    GEOSGeom_getYMin_r,
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





# initialize the numpy API
np.import_array()

# initialize PyGEOS C API
import_pygeos_c_api()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_parts(object[:] array):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t idx = 0
    cdef GEOSGeometry *geom = NULL
    cdef const GEOSGeometry *part = NULL

    counts = pygeos.get_num_geometries(array)

    # None elements in array return -1 for count, so
    # they must be filtered out before calculating total count
    cdef Py_ssize_t count = counts[counts>0].sum()

    if count <= 0:
        # return immediately if there are no geometries to return
        # count is negative when the only entries in array are None
        return (
            np.empty(shape=(0, ), dtype=np.object),
            np.empty(shape=(0, ), dtype=np.intp)
        )

    parts = np.empty(shape=(count, ), dtype=np.object)
    index = np.empty(shape=(count, ), dtype=np.intp)

    cdef int[:] counts_view = counts
    cdef object[:] parts_view = parts
    cdef np.intp_t[:] index_view = index

    with get_geos_handle() as geos_handle:
        for geom_idx in range(array.size):
            if counts_view[geom_idx] <= 0:
                # No parts to return, skip this item
                continue

            if PyGEOS_GetGEOSGeometry(<PyObject *>array[geom_idx], &geom) == 0:
                raise TypeError("One of the arguments is of incorrect type. "
                                "Please provide only Geometry objects.")

            if geom == NULL:
                continue

            for part_idx in range(counts_view[geom_idx]):
                index_view[idx] = geom_idx
                part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)

                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                # cast part back to <GEOSGeometry> to discard const qualifier
                # pending issue #227
                parts_view[idx] = PyGEOS_CreateGeometry(<GEOSGeometry *>part, geos_handle)

                idx += 1

    return parts, index



cdef GEOSGeometry* create_box(GEOSContextHandle_t geos_handle,
                              double xmin, double ymin, double xmax, double ymax) nogil:

    cdef GEOSCoordSequence *coords = NULL
    cdef GEOSGeometry *geom = NULL

# try:
    # create polygon starting at bottom left oriented  counterclockwise
    coords = GEOSCoordSeq_create_r(geos_handle, 5, 2)
    GEOSCoordSeq_setXY_r(geos_handle, coords, 0, xmin, ymin)
    GEOSCoordSeq_setXY_r(geos_handle, coords, 1, xmax, ymin)
    GEOSCoordSeq_setXY_r(geos_handle, coords, 2, xmax, ymax)
    GEOSCoordSeq_setXY_r(geos_handle, coords, 3, xmin, ymax)
    GEOSCoordSeq_setXY_r(geos_handle, coords, 4, xmin, ymin)

    # construct linear ring then construct polygon
    # NOTE: coords then ring become owned by polygon
    # and are not to be cleaned up here
    ring = GEOSGeom_createLinearRing_r(geos_handle, coords)
    geom = GEOSGeom_createPolygon_r(geos_handle, ring, NULL, 0)

    # except:
    #     if coords != NULL:
    #         GEOSCoordSeq_destroy_r(geos_handle, coords)

    #     if geom != NULL:
    #         GEOSGeom_destroy_r(geos_handle, geom)

    return geom


cdef int get_bounds(GEOSContextHandle_t geos_handle,
                    const GEOSGeometry *geom,
                    double *xmin, double *ymin, double *xmax, double *ymax) nogil except 0:

    # TODO: fails if empty / NULL, return 0

    # NOTE: limited to GEOS >= 3.7
    GEOSGeom_getXMin_r(geos_handle, geom, xmin)
    GEOSGeom_getYMin_r(geos_handle, geom, ymin)
    GEOSGeom_getXMax_r(geos_handle, geom, xmax)
    GEOSGeom_getYMax_r(geos_handle, geom, ymax)

    return 1

# last params: resizeable geom vector
# returns count of geometries
# TODO: depth can be 8 bit uint
# max_vertices maybe uint32, and check bounds on input below
cdef int _subdivide_geometry(GEOSContextHandle_t geos_handle,
                             const GEOSGeometry *geom,
                             int geom_dimension, int max_vertices, uint8_t depth,
                             void *out_geom_vec) nogil:
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t geom_idx = 0
    cdef uint64_t count = 0
    cdef int num_parts = 0
    cdef int num_coords = 0
    cdef double xmax = DBL_MIN
    cdef double ymax = DBL_MIN
    cdef double xmin = DBL_MAX
    cdef double ymin = DBL_MAX
    cdef Py_ssize_t quadrant = 0
    cdef double center_x = DBL_MAX
    cdef double center_y = DBL_MAX
    cdef GEOSGeometry *clip_geom = NULL
    cdef const GEOSGeometry *part = NULL
    cdef const GEOSGeometry *clipped_geom = NULL

    # TODO: consolidate break conditions:
    # hit recursion limit
    # input is null, empty, point, or has < max_vertices (even if multi?)

    if depth > 50:
        # TODO: add better comment
        out_geom = GEOSGeom_clone_r(geos_handle, geom)
        (<GeometryVector>out_geom_vec).push(out_geom)
        return 1

    depth += 1

    cdef int type_id = GEOSGeomTypeId_r(geos_handle, geom)

    # if empty or singular point, return copy
    if GEOSisEmpty_r(geos_handle, geom) or type_id == 0:
        out_geom = GEOSGeom_clone_r(geos_handle, geom)
        (<GeometryVector>out_geom_vec).push(out_geom)
        return 1

    # DOCUMENTATION NOTE: this will also strip any lower-dimension input geometries from
    # input geometries
    if GEOSGeom_getDimensions_r(geos_handle, geom) < geom_dimension:
        return 0

    # Multi* or GeometryCollection: recurse over parts
    if type_id >= 4:
        num_parts = GEOSGetNumGeometries_r(geos_handle, geom)

        for part_idx in range(num_parts):
            part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)
            if part == NULL:
                # if passed in from top-level we may want to add to vector and return
                # if results from clip, we want to discard
                pass

            else:
                count += _subdivide_geometry(geos_handle, part, geom_dimension, max_vertices, depth, out_geom_vec)

        return count

    # count points
    num_coords = GEOSGetNumCoordinates_r(geos_handle, geom)

    if num_coords <= max_vertices:
        # done splitting, return
        out_geom = GEOSGeom_clone_r(geos_handle, geom)
        (<GeometryVector>out_geom_vec).push(out_geom)
        return 1

    get_bounds(geos_handle, geom, &xmin, &ymin, &xmax, &ymax)
    # printf("bounds: %.0f, %.0f, %.0f, %.0f\n", xmin, ymin, xmax, ymax)

    if (xmax - xmin) == 0 or (ymax - ymin) == 0:
        # if bounds collapse to a point or a line, can't subdivide further
        # TODO: figure out when this might happen, is it valid if input is a vertical
        # horizontal line?
        return 0

    # split into 4 tiles from center of bounds
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # try:
    for quadrant in range(4):
        if quadrant == 0:  # bottom left
            clip_geom = create_box(geos_handle, xmin, ymin, center_x, center_y)

        elif quadrant == 1:  # top left
            clip_geom = create_box(geos_handle, xmin, center_y, center_x, ymax)

        elif quadrant == 2:  # top right
            clip_geom = create_box(geos_handle, center_x, center_y, xmax, ymax)

        else:  # bottom right
            clip_geom = create_box(geos_handle, center_x, ymin, xmax, center_y)

        # clip by clip_geom
        # TODO: use precision version (what GEOS version?)
        # TODO: if null
        clipped_geom = GEOSIntersection_r(geos_handle, geom, clip_geom)

        # simplify geometry
        # TODO: if NULL
        clipped_geom = GEOSSimplify_r(geos_handle, clipped_geom, 0)

        if GEOSisEmpty_r(geos_handle, clipped_geom):
            # printf("Clipped geom is empty, skip\n")
            continue

        # check if collapsed dimensions
        # NOTE: this doesn't catch geometry collections that include lower dimensions
        if GEOSGeom_getDimensions_r(geos_handle, clipped_geom) < GEOSGeom_getDimensions_r(geos_handle, geom):
            # printf("Clip collapsed dimension, skip\n")
            continue

        count += _subdivide_geometry(geos_handle, clipped_geom, geom_dimension, max_vertices, depth, out_geom_vec)

        GEOSGeom_destroy_r(geos_handle, clipped_geom)
        GEOSGeom_destroy_r(geos_handle, clip_geom)


    # TODO: how to cleanup if not using try / finally due to gil?
    # finally:
    # if clip_geom != NULL:
    #     GEOSGeom_destroy_r(geos_handle, clip_geom)

    return count


# TODO: decision: filter to only geometries (filter out empty / NULL)?
# TODO: crosscheck ST_subdivide results
def subdivide(object[:] array, int max_vertices=100):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t idx = 0
    cdef int geom_dimension = 0;
    cdef GEOSGeometry *geom = NULL
    cdef Py_ssize_t[:] out_idx2

    if max_vertices <= 5:
        raise ValueError("max_vertices must be greater than 5")

    cdef Py_ssize_t reserved_size = len(array)

    with get_geos_handle() as geos_handle, \
         GeometryVector(reserved_size) as out_geom_vec, \
         IndexVector(reserved_size) as out_idx_vec:

        for geom_idx in range(array.size):
            if PyGEOS_GetGEOSGeometry(<PyObject *>array[geom_idx], &geom) == 0:
                raise TypeError("One of the arguments is of incorrect type. "
                                "Please provide only Geometry objects.")

            if geom == NULL:
                # input was None
                out_geom_vec.push(NULL)
                out_idx_vec.push(geom_idx)
                continue

            geom_dimension = GEOSGeom_getDimensions_r(geos_handle, geom)

            count = _subdivide_geometry(geos_handle, <const GEOSGeometry*>geom, geom_dimension, max_vertices, 0, <void*>out_geom_vec)
            for i in range(count):
                out_idx_vec.push(geom_idx)

        out_idx = out_idx_vec.to_array()
        out_geom = out_geom_vec.to_array(geos_handle)

        return out_geom, out_idx

