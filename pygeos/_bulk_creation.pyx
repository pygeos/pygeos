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
    GEOSGeomTypeId_r,
    get_geos_handle,
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
def collections_1d(
    object geometries,
    object indices,
    int geometry_type = 7,
    int ndim = 2
):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t coll_idx = 0
    cdef Py_ssize_t coll_size = 0
    cdef Py_ssize_t first_geom_idx = 0
    cdef Py_ssize_t this_geom_idx = 0
    cdef Py_ssize_t n_missing = 0
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *coll = NULL
    cdef int expected_type = -1
    cdef int expected_type_alt = -1
    cdef int curr_type = -1

    if geometry_type == 4:  # MULTIPOINT
        expected_type = 0
    elif geometry_type == 5:  # MULTILINESTRING
        expected_type = 1
        expected_type_alt = 2
    elif geometry_type == 6:  # MULTIPOLYGON
        expected_type = 3
    elif geometry_type == 7:
        pass
    else:
        raise ValueError(f"Invalid geometry_type: {geometry_type}.")

    # Cast input arrays and define memoryviews for later usage
    geometries = np.asarray(geometries, dtype=np.object)
    if geometries.ndim != 1:
        raise TypeError("geometries is not a one-dimensional array.")

    indices = np.asarray(indices, dtype=np.int32)
    if indices.ndim != 1:
        raise TypeError("indices is not a one-dimensional array.")

    if geometries.shape[0] != indices.shape[0]:
        raise ValueError("geometries and indices do not have equal size.")

    if geometries.shape[0] == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=np.object_)

    if np.any(indices[1:] < indices[:-1]):
        raise ValueError("The indices should be sorted.")  

    cdef object[:] geometries_view = geometries
    cdef int[:] indices_view = indices

    # get the geometry count per collection
    cdef long[:] collection_size = np.bincount(indices)

    # A temporary array for the geometries that will be given to CreateCollection.
    # Its size equals max(collection_size) to accomodate the largest collection.
    temp_geoms = np.empty(shape=(np.max(collection_size), ), dtype=np.intp)
    cdef np.intp_t[:] temp_geoms_view = temp_geoms

    # The final target array
    cdef Py_ssize_t n_colls = collection_size.shape[0]
    result = np.empty(shape=(n_colls, ), dtype=np.object_)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for coll_idx in range(n_colls):
            coll_size = collection_size[coll_idx]

            # fill the temporary array with geometries belonging to this collection
            for this_geom_idx in range(coll_size):
                geom_idx = first_geom_idx + this_geom_idx
                if PyGEOS_GetGEOSGeometry(<PyObject *>geometries_view[geom_idx], &geom) == 0:
                    # deallocate previous temp geometries (preventing memory leaks)
                    for geom_idx in range(this_geom_idx):
                        GEOSGeom_destroy_r(geos_handle, <GEOSGeometry *>temp_geoms_view[geom_idx])
                    raise TypeError(
                        "One of the arguments is of incorrect type. Please provide only Geometry objects."
                    )

                # Check geometry subtype for non-geometrycollections
                if geometry_type != 7:
                    curr_type = GEOSGeomTypeId_r(geos_handle, geom)
                    if curr_type != expected_type and curr_type != expected_type_alt:
                        # deallocate previous temp geometries (preventing memory leaks)
                        for geom_idx in range(this_geom_idx):
                            GEOSGeom_destroy_r(geos_handle, <GEOSGeometry *>temp_geoms_view[geom_idx])
                        raise TypeError(
                            f"One of the arguments has unexpected geometry type {curr_type}."
                        )

                # ignore missing values
                if geom == NULL:
                    n_missing += 1
                else:
                    # assign to the temporary geometry array
                    temp_geoms_view[this_geom_idx] = <np.intp_t>GEOSGeom_clone_r(geos_handle, geom)

            # create the collection
            coll = GEOSGeom_createCollection_r(
                geos_handle,
                geometry_type, 
                <GEOSGeometry**> &temp_geoms_view[0],
                <unsigned int>(coll_size - n_missing)
            )

            result_view[coll_idx] = PyGEOS_CreateGeometry(
                coll, geos_handle
            )
            first_geom_idx += coll_size
            n_missing = 0

    return result
