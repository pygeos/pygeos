from cpython cimport PyObject
cimport cython

import numpy as np
cimport numpy as np

from ._geos cimport GEOS_init_r, GEOS_finish_r, GEOSContextHandle_t, GEOSGeometry, GEOSGetNumGeometries_r
from ._pygeos_lib cimport GeometryObject, get_geom


@cython.boundscheck(False)
@cython.wraparound(False)
def geos_get_num_geometries(object[:] array):
    cdef unsigned int i = 0
    cdef GEOSContextHandle_t ctx = GEOS_init_r()
    cdef GEOSGeometry *geom = NULL
    cdef GeometryObject *obj

    counts = np.zeros(shape=(array.size), dtype=np.intp)
    cdef np.intp_t [:] counts_view = counts[:]

    for i in range(array.size):
        # This does not work because get_geom checks the type against _pygeos_lib.GeometryObject
        # while the geometry is of type ._geos.GeometryObject. These are defined in the same
        # source file, but compiled into different shared libraries, so their memory addresses
        # are not the same.
        #
        # if get_geom(<GeometryObject *>array[i], &geom) == 0:
        #     raise TypeError("One of the arguments is of incorrect type. Please provide only Geometry objects.")

        # if geom == NULL:
        #     continue
        # 
        # counts_view[i] = GEOSGetNumGeometries_r(ctx, geom)

        # This works but may segfault:
        obj = <GeometryObject *>array[i]
        counts_view[i] = GEOSGetNumGeometries_r(ctx, <GEOSGeometry *>obj.ptr)

    GEOS_finish_r(ctx)
    return counts
