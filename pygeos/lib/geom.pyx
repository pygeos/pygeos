from cpython cimport PyObject
cimport cython

import numpy as np
cimport numpy as np

from pygeos.lib.geos_wrapper cimport (
    GEOSContextHandle_t,
    GEOSGeometry,
    GEOSGeom_clone_r,
    GEOSGetGeometryN_r,
    GEOSGetNumGeometries_r,
    GEOS_init_r
)
from pygeos.lib.pygeos_wrapper cimport (
    import_pygeos_core_api,
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry
)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef geos_get_num_geometries(object[:] array):
    cdef Py_ssize_t i = 0
    cdef GEOSContextHandle_t geos_handle = GEOS_init_r()
    cdef GEOSGeometry *geom = NULL

    counts = np.zeros(shape=(array.size), dtype=np.intp)
    cdef np.intp_t[:] counts_view = counts[:]

    for i in range(array.size):
        if PyGEOS_GetGEOSGeometry(<PyObject *>array[i], &geom) == 0:
            raise TypeError("One of the arguments is of incorrect type. Please provide "
            "only Geometry objects.")

        if geom == NULL:
            continue

        counts_view[i] = GEOSGetNumGeometries_r(geos_handle, geom)

    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
def get_parts(object[:] array):
    import_pygeos_core_api()

    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t idx = 0
    cdef GEOSContextHandle_t geos_handle = GEOS_init_r()
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *part = NULL

    cdef const np.intp_t [:] input_index_view = np.arange(0, len(array), dtype=np.intp)

    counts = geos_get_num_geometries(array)
    cdef np.intp_t [:] counts_view = counts[:]

    cdef Py_ssize_t count = counts.sum()

    parts = np.empty(shape=(count, ), dtype=np.object)
    index = np.empty(shape=(count, ), dtype=np.intp)

    cdef object[:] parts_view = parts[:]
    cdef np.intp_t [:] index_view = index[:]

    for geom_idx in range(array.size):
        if PyGEOS_GetGEOSGeometry(<PyObject *>array[geom_idx], &geom) == 0:
            raise TypeError("One of the arguments is of incorrect type. Please provide "
            "only Geometry objects.")

        if geom == NULL:
            continue

        for part_idx in range(counts_view[geom_idx]):
            index_view[idx] = geom_idx
            part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)

            if part == NULL:
                parts_view[idx] = None

            else:
                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                parts_view[idx] = <object>PyGEOS_CreateGeometry(part, geos_handle)

            idx += 1


    return parts, index
