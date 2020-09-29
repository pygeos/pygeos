from cpython cimport PyObject
cimport cython

import numpy as np
cimport numpy as np


from pygeos.lib.geos_wrapper cimport *
from pygeos.lib.pygeos_wrapper cimport *


@cython.boundscheck(False)
@cython.wraparound(False)
cdef geos_get_num_geometries(object[:] array):
    cdef unsigned int i = 0
    cdef GEOSContextHandle_t geos_handle = get_geos_handle()
    cdef GEOSGeometry *geom = NULL

    counts = np.zeros(shape=(array.size), dtype=np.intp)
    cdef np.intp_t [:] counts_view = counts[:]

    for i in range(array.size):
        PyGEOSGetGEOSGeom(<GeometryObject *>array[i], &geom)

        if geom == NULL or GEOSisEmpty_r(geos_handle, geom):
            continue

        counts_view[i] = GEOSGetNumGeometries_r(geos_handle, geom)

    return counts


@cython.boundscheck(False)
@cython.wraparound(False)
def get_parts(object[:] array):
    import_pygeos_core_api()

    cdef unsigned geom_idx = 0
    cdef unsigned part_idx = 0
    cdef unsigned idx = 0
    cdef GEOSContextHandle_t geos_handle = get_geos_handle()
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *part = NULL

    cdef const np.intp_t [:] input_index_view = np.arange(0, len(array))

    counts = geos_get_num_geometries(array)
    cdef np.intp_t [:] counts_view = counts[:]

    cdef unsigned int count = counts.sum()

    parts = np.empty(shape=(count, ), dtype=np.object)
    index = np.empty(shape=(count, ), dtype=np.intp)

    cdef object[:] parts_view = parts[:]
    cdef np.intp_t [:] index_view = index[:]

    for geom_idx in range(array.size):
        PyGEOSGetGEOSGeom(<GeometryObject *>array[geom_idx], &geom)

        if geom == NULL or GEOSisEmpty_r(geos_handle, geom):
            continue

        for part_idx in range(counts_view[geom_idx]):

            index_view[idx] = geom_idx
            part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)

            if part == NULL:
                parts_view[idx] = None

            else:
                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                parts_view[idx] = <object>PyGEOSCreateGeom(part, geos_handle)

            idx += 1


    return parts, index
