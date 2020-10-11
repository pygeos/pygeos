from cpython cimport PyObject
cimport cython

import numpy as np
cimport numpy as np

from ._geos cimport GEOS_init_r, GEOS_finish_r, GEOSContextHandle_t, GEOSGeometry, GEOSGetNumGeometries_r
from ._pygeos_lib cimport PyGEOS_GetGeom, import_pygeos_api


import_pygeos_api()


@cython.boundscheck(False)
@cython.wraparound(False)
def geos_get_num_geometries(object[:] array):
    cdef unsigned int i = 0
    cdef GEOSContextHandle_t ctx = GEOS_init_r()
    cdef GEOSGeometry *geom = NULL

    counts = np.zeros(shape=(array.size), dtype=np.intp)
    cdef np.intp_t [:] counts_view = counts[:]

    for i in range(array.size):
        if PyGEOS_GetGeom(<PyObject *>array[i], &geom) == 0:
            raise TypeError("One of the arguments is of incorrect type. Please provide only Geometry objects.")

        if geom == NULL:
            continue
        
        counts_view[i] = GEOSGetNumGeometries_r(ctx, geom)

    GEOS_finish_r(ctx)
    return counts
