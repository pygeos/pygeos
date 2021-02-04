# distutils: define_macros=GEOS_USE_ONLY_R_API

from cpython cimport PyObject
from cython cimport view

cimport cython

import numpy as np
cimport numpy as np
import pygeos

from pygeos._geos cimport (
    GEOSGeometry,
    GEOSGeom_destroy_r,
    GEOSGeom_clone_r,
    GEOSGeom_createCollection_r,
    get_geos_handle
)
from pygeos._pygeos_api cimport (
    import_pygeos_c_api,
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry
)

# initialize PyGEOS C API
import_pygeos_c_api()


@cython.boundscheck(False)
@cython.wraparound(False)
def collections_1d(object geometries, object indices, int geom_type = 7, int ndim = 2):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t coll_idx = 0
    cdef Py_ssize_t coll_size = 0
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *coll = NULL

    cdef object[:] geometries_view = np.asarray(geometries, dtype=np.object)
    cdef int[:] indices_view = np.asarray(indices, dtype=np.int32)

    if geometries_view.size != indices_view.size:
        raise ValueError("Geometries and indices array should have equal size.")

    cdef Py_ssize_t n_geoms = geometries_view.size
    cdef Py_ssize_t n_colls = indices_view[indices_view.size - 1] + 1
    

    if n_geoms == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=np.object_)

    # A temporary array for the geometries that will be given to CreateCollection.
    # Its size equals n_geoms, which is much too large in most cases. But it will
    # give overhead to investigate the optimal size at this point.
    temp_geoms = np.empty(shape=(n_geoms, ), dtype=np.intp)
    cdef np.intp_t[:] temp_geoms_view = temp_geoms

    # The final target array
    result = np.empty(shape=(n_colls, ), dtype=np.object_)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for coll_idx in range(n_colls):
            coll_size = 0

            # fill the temporary array with geometries belonging to this collection
            for geom_idx in range(geom_idx, n_geoms):
                if indices_view[geom_idx] != coll_idx:
                    break

                if PyGEOS_GetGEOSGeometry(<PyObject *>geometries_view[geom_idx], &geom) == 0:
                    # deallocate previous temp geometries (preventing memory leaks)
                    for geom_idx in range(coll_size):
                        GEOSGeom_destroy_r(geos_handle, <GEOSGeometry *>temp_geoms_view[geom_idx])
                    raise TypeError(
                        "One of the arguments is of incorrect type. Please provide only Geometry objects."
                    )

                # ignore missing values
                if geom == NULL:
                    continue

                # assign to the temporary geometry array
                temp_geoms_view[coll_size] = <np.intp_t>GEOSGeom_clone_r(geos_handle, geom)
                coll_size += 1

            # create the collection
            coll = GEOSGeom_createCollection_r(
                geos_handle,
                geom_type, 
                <GEOSGeometry**> &temp_geoms_view[0],
                <unsigned int>coll_size
            )

            result_view[coll_idx] = PyGEOS_CreateGeometry(
                coll, geos_handle
            )

    return result
