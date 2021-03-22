# distutils: define_macros=GEOS_USE_ONLY_R_API

from cpython cimport PyObject
cimport cython
from cython cimport view

import numpy as np
cimport numpy as np
import pygeos

from pygeos._geos cimport (
    GEOSGeometry,
    GEOSCoordSequence,
    GEOSGeom_clone_r,
    GEOSGetGeometryN_r,
    get_geos_handle,
    GEOSGeom_destroy_r,
    GEOSGeom_createCollection_r,
    GEOSGeomTypeId_r,
    GEOSCoordSeq_create_r,
    GEOSCoordSeq_setX_r,
    GEOSCoordSeq_setY_r,
    GEOSCoordSeq_setZ_r,
    GEOSGeom_createPoint_r,
    GEOSGeom_createLineString_r,
    GEOSGeom_createLinearRing_r
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
def simple_geometries_1d(object coordinates, object indices, int geometry_type):
    cdef Py_ssize_t idx = 0
    cdef unsigned int coord_idx = 0
    cdef Py_ssize_t geom_idx = 0
    cdef unsigned int geom_size = 0
    cdef unsigned int ring_closure = 0
    cdef Py_ssize_t coll_geom_idx = 0
    cdef char include_z
    cdef GEOSGeometry *geom = NULL
    cdef GEOSCoordSequence *seq = NULL

    # Cast input arrays and define memoryviews for later usage
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2:
        raise TypeError("coordinates is not a two-dimensional array.")

    indices = np.asarray(indices, dtype=np.intp)
    if indices.ndim != 1:
        raise TypeError("indices is not a one-dimensional array.")

    if coordinates.shape[0] != indices.shape[0]:
        raise ValueError("geometries and indices do not have equal size.")

    cdef unsigned int dims = coordinates.shape[1]
    if dims not in {2, 3}:
        raise ValueError("coordinates should N by 2 or N by 3.")

    if geometry_type not in {0, 1, 2}:
        raise ValueError(f"Invalid geometry_type: {geometry_type}.")

    if coordinates.shape[0] == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=np.object_)

    if np.any(indices[1:] < indices[:indices.shape[0] - 1]):
        raise ValueError("The indices should be sorted.")  

    cdef double[:, :] coord_view = coordinates
    cdef np.intp_t[:] index_view = indices

    # get the geometry count per collection
    cdef unsigned int[:] coord_counts = np.bincount(indices).astype(np.uint32)

    # The final target array
    cdef Py_ssize_t n_geoms = coord_counts.shape[0]
    result = np.empty(shape=(n_geoms, ), dtype=object)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for geom_idx in range(n_geoms):
            geom_size = coord_counts[geom_idx]

            # check if we need to close a linearring
            if geometry_type == 2:
                ring_closure = 0
                for coord_idx in range(dims):
                    if coord_view[idx, coord_idx] != coord_view[idx + geom_size - 1, coord_idx]:
                        ring_closure = 1
                        break

            seq = GEOSCoordSeq_create_r(geos_handle, geom_size + ring_closure, dims)
            for coord_idx in range(geom_size):
                if GEOSCoordSeq_setX_r(geos_handle, seq, coord_idx, coord_view[idx, 0]) == 0:
                    return
                if GEOSCoordSeq_setY_r(geos_handle, seq, coord_idx, coord_view[idx, 1]) == 0:
                    return
                if dims == 3:
                    if GEOSCoordSeq_setZ_r(geos_handle, seq, coord_idx, coord_view[idx, 2]) == 0:
                        return
                idx += 1

            if geometry_type == 0:
                geom = GEOSGeom_createPoint_r(geos_handle, seq)
            elif geometry_type == 1:
                geom = GEOSGeom_createLineString_r(geos_handle, seq)
            elif geometry_type == 2:
                if ring_closure == 1:
                    if GEOSCoordSeq_setX_r(geos_handle, seq, geom_size, coord_view[idx - geom_size, 0]) == 0:
                        return
                    if GEOSCoordSeq_setY_r(geos_handle, seq, geom_size, coord_view[idx - geom_size, 1]) == 0:
                        return
                    if dims == 3:
                        if GEOSCoordSeq_setZ_r(geos_handle, seq, geom_size, coord_view[idx - geom_size, 2]) == 0:
                            return
                geom = GEOSGeom_createLinearRing_r(geos_handle, seq)

            if geom == NULL:
                return

            result_view[geom_idx] = PyGEOS_CreateGeometry(geom, geos_handle)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def get_parts(object[:] array):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t idx = 0
    cdef GEOSGeometry *geom = NULL
    cdef const GEOSGeometry *part = NULL

    counts = pygeos.get_num_geometries(array)
    cdef Py_ssize_t count = counts.sum()

    if count == 0:
        # return immediately if there are no geometries to return
        return (
            np.empty(shape=(0, ), dtype=object),
            np.empty(shape=(0, ), dtype=np.intp)
        )

    parts = np.empty(shape=(count, ), dtype=object)
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
                if part == NULL:
                    return

                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                if part == NULL:
                    return
                # cast part back to <GEOSGeometry> to discard const qualifier
                # pending issue #227
                parts_view[idx] = PyGEOS_CreateGeometry(<GEOSGeometry *>part, geos_handle)

                idx += 1

    return parts, index


cdef _deallocate_arr(void* handle, np.intp_t[:] arr, Py_ssize_t last_geom_i):
    """Deallocate a temporary geometry array to prevent memory leaks"""
    cdef Py_ssize_t i = 0

    for i in range(last_geom_i):
        GEOSGeom_destroy_r(handle, <GEOSGeometry *>arr[i])


@cython.boundscheck(False)
@cython.wraparound(False)
def collections_1d(object geometries, object indices, int geometry_type = 7):
    cdef Py_ssize_t geom_idx_1 = 0
    cdef Py_ssize_t coll_idx = 0
    cdef unsigned int coll_size = 0
    cdef Py_ssize_t coll_geom_idx = 0
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
    geometries = np.asarray(geometries, dtype=object)
    if geometries.ndim != 1:
        raise TypeError("geometries is not a one-dimensional array.")

    indices = np.asarray(indices, dtype=np.int32)
    if indices.ndim != 1:
        raise TypeError("indices is not a one-dimensional array.")

    if geometries.shape[0] != indices.shape[0]:
        raise ValueError("geometries and indices do not have equal size.")

    if geometries.shape[0] == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=object)

    if np.any(indices[1:] < indices[:indices.shape[0] - 1]):
        raise ValueError("The indices should be sorted.")  

    cdef object[:] geometries_view = geometries
    cdef int[:] indices_view = indices

    # get the geometry count per collection
    cdef int[:] collection_size = np.bincount(indices).astype(np.int32)

    # A temporary array for the geometries that will be given to CreateCollection.
    # Its size equals max(collection_size) to accomodate the largest collection.
    temp_geoms = np.empty(shape=(np.max(collection_size), ), dtype=np.intp)
    cdef np.intp_t[:] temp_geoms_view = temp_geoms

    # The final target array
    cdef Py_ssize_t n_colls = collection_size.shape[0]
    result = np.empty(shape=(n_colls, ), dtype=object)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for coll_idx in range(n_colls):
            coll_size = 0

            # fill the temporary array with geometries belonging to this collection
            for coll_geom_idx in range(collection_size[coll_idx]):
                if PyGEOS_GetGEOSGeometry(<PyObject *>geometries_view[geom_idx_1 + coll_geom_idx], &geom) == 0:
                    _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                    raise TypeError(
                        "One of the arguments is of incorrect type. Please provide only Geometry objects."
                    )

                # ignore missing values
                if geom == NULL:
                    continue

                # Check geometry subtype for non-geometrycollections
                if geometry_type != 7:
                    curr_type = GEOSGeomTypeId_r(geos_handle, geom)
                    if curr_type == -1:
                        _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                        return
                    if curr_type != expected_type and curr_type != expected_type_alt:
                        _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                        raise TypeError(
                            f"One of the arguments has unexpected geometry type {curr_type}."
                        )

                # assign to the temporary geometry array  
                geom = GEOSGeom_clone_r(geos_handle, geom)
                if geom == NULL:
                    _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                    return                 
                temp_geoms_view[coll_size] = <np.intp_t>geom
                coll_size += 1

            # create the collection
            coll = GEOSGeom_createCollection_r(
                geos_handle,
                geometry_type, 
                <GEOSGeometry**> &temp_geoms_view[0],
                coll_size
            )
            if coll == NULL:
                return

            result_view[coll_idx] = PyGEOS_CreateGeometry(coll, geos_handle)

            geom_idx_1 += collection_size[coll_idx]

    return result
