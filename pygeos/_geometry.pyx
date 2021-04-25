# distutils: define_macros=GEOS_USE_ONLY_R_API

cimport cython
from cpython cimport PyObject
from cython cimport view

import numpy as np

cimport numpy as np

import pygeos

from pygeos._geos cimport (
    GEOSContextHandle_t,
    GEOSCoordSeq_create_r,
    GEOSCoordSeq_destroy_r,
    GEOSCoordSeq_setX_r,
    GEOSCoordSeq_setY_r,
    GEOSCoordSeq_setZ_r,
    GEOSCoordSequence,
    GEOSGeom_clone_r,
    GEOSGeom_createCollection_r,
    GEOSGeom_createLinearRing_r,
    GEOSGeom_createLineString_r,
    GEOSGeom_createPoint_r,
    GEOSGeom_createPolygon_r,
    GEOSGeom_destroy_r,
    GEOSGeometry,
    GEOSGeomTypeId_r,
    GEOSGetGeometryN_r,
    GEOSGetExteriorRing_r,
    GEOSGetInteriorRingN_r,
    get_geos_handle,
)
from pygeos._pygeos_api cimport (
    import_pygeos_c_api,
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry,
)

# initialize PyGEOS C API
import_pygeos_c_api()


cdef char _set_xyz(GEOSContextHandle_t geos_handle, GEOSCoordSequence *seq, unsigned int coord_idx,
                   unsigned int dims, double[:, :] coord_view, Py_ssize_t idx):
    if GEOSCoordSeq_setX_r(geos_handle, seq, coord_idx, coord_view[idx, 0]) == 0:
        return 0
    if GEOSCoordSeq_setY_r(geos_handle, seq, coord_idx, coord_view[idx, 1]) == 0:
        return 0
    if dims == 3:
        if GEOSCoordSeq_setZ_r(geos_handle, seq, coord_idx, coord_view[idx, 2]) == 0:
            return 0
    return 1

 
@cython.boundscheck(False)
@cython.wraparound(False)
def simple_geometries_1d(object coordinates, object indices, int geometry_type):
    cdef Py_ssize_t idx = 0
    cdef unsigned int coord_idx = 0
    cdef Py_ssize_t geom_idx = 0
    cdef unsigned int geom_size = 0
    cdef unsigned int ring_closure = 0
    cdef Py_ssize_t coll_geom_idx = 0
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
        raise ValueError("The indices must be sorted.")  

    cdef double[:, :] coord_view = coordinates
    cdef np.intp_t[:] index_view = indices

    # get the geometry count per collection (this raises on negative indices)
    cdef unsigned int[:] coord_counts = np.bincount(indices).astype(np.uint32)

    # The final target array
    cdef Py_ssize_t n_geoms = coord_counts.shape[0]
    result = np.empty(shape=(n_geoms, ), dtype=object)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for geom_idx in range(n_geoms):
            geom_size = coord_counts[geom_idx]

            # insert None if there are no coordinates
            if geom_size == 0:
                result_view[geom_idx] = PyGEOS_CreateGeometry(NULL, geos_handle)
                continue

            # check if we need to close a linearring
            if geometry_type == 2:
                ring_closure = 0
                for coord_idx in range(dims):
                    if coord_view[idx, coord_idx] != coord_view[idx + geom_size - 1, coord_idx]:
                        ring_closure = 1
                        break

            seq = GEOSCoordSeq_create_r(geos_handle, geom_size + ring_closure, dims)
            for coord_idx in range(geom_size):
                if _set_xyz(geos_handle, seq, coord_idx, dims, coord_view, idx) == 0:
                    GEOSCoordSeq_destroy_r(geos_handle, seq)
                    return  # GEOSException is raised by get_geos_handle
                idx += 1

            if geometry_type == 0:
                geom = GEOSGeom_createPoint_r(geos_handle, seq)
            elif geometry_type == 1:
                geom = GEOSGeom_createLineString_r(geos_handle, seq)
            elif geometry_type == 2:
                if ring_closure == 1:
                    if _set_xyz(geos_handle, seq, geom_size, dims, coord_view, idx - geom_size) == 0:
                        GEOSCoordSeq_destroy_r(geos_handle, seq)
                        return  # GEOSException is raised by get_geos_handle
                geom = GEOSGeom_createLinearRing_r(geos_handle, seq)

            if geom == NULL:
                return  # GEOSException is raised by get_geos_handle

            result_view[geom_idx] = PyGEOS_CreateGeometry(geom, geos_handle)

    return result



cdef const GEOSGeometry* GetRingN(GEOSContextHandle_t handle, GEOSGeometry* polygon, int n):
    if n == 0:
        return GEOSGetExteriorRing_r(handle, polygon)
    else:
        return GEOSGetInteriorRingN_r(handle, polygon, n - 1)



@cython.boundscheck(False)
@cython.wraparound(False)
def get_parts(object[:] array, bint extract_rings=0):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t count
    cdef GEOSGeometry *geom = NULL
    cdef const GEOSGeometry *part = NULL

    if extract_rings:
        counts = pygeos.get_num_interior_rings(array)
        is_polygon = (pygeos.get_type_id(array) == 3) & (~pygeos.is_empty(array))
        counts += is_polygon
        count = counts.sum()
    else:
        counts = pygeos.get_num_geometries(array)
        count = counts.sum()

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

                if extract_rings:
                    part = GetRingN(geos_handle, geom, part_idx)
                else:
                    part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)
                if part == NULL:
                    return  # GEOSException is raised by get_geos_handle

                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                if part == NULL:
                    return  # GEOSException is raised by get_geos_handle

                # cast part back to <GEOSGeometry> to discard const qualifier
                # pending issue #227
                parts_view[idx] = PyGEOS_CreateGeometry(<GEOSGeometry *>part, geos_handle)

                idx += 1

    return parts, index


cdef _deallocate_arr(void* handle, np.intp_t[:] arr, Py_ssize_t last_geom_i):
    """Deallocate a temporary geometry array to prevent memory leaks"""
    cdef Py_ssize_t i = 0
    cdef GEOSGeometry *g

    for i in range(last_geom_i):
        g = <GEOSGeometry *>arr[i]
        if g != NULL:
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

    # get the geometry count per collection (this raises on negative indices)
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
                        return  # GEOSException is raised by get_geos_handle
                    if curr_type != expected_type and curr_type != expected_type_alt:
                        _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                        raise TypeError(
                            f"One of the arguments has unexpected geometry type {curr_type}."
                        )

                # assign to the temporary geometry array  
                geom = GEOSGeom_clone_r(geos_handle, geom)
                if geom == NULL:
                    _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                    return  # GEOSException is raised by get_geos_handle           
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
                return  # GEOSException is raised by get_geos_handle

            result_view[coll_idx] = PyGEOS_CreateGeometry(coll, geos_handle)

            geom_idx_1 += collection_size[coll_idx]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def polygons_1d(object shells, object holes, object indices):
    cdef Py_ssize_t hole_idx_1 = 0
    cdef Py_ssize_t poly_idx = 0
    cdef unsigned int n_holes = 0
    cdef int geom_type = 0
    cdef Py_ssize_t poly_hole_idx = 0
    cdef GEOSGeometry *shell = NULL
    cdef GEOSGeometry *hole = NULL
    cdef GEOSGeometry *poly = NULL

    # Cast input arrays and define memoryviews for later usage
    shells = np.asarray(shells, dtype=object)
    if shells.ndim != 1:
        raise TypeError("shells is not a one-dimensional array.")

    # Cast input arrays and define memoryviews for later usage
    holes = np.asarray(holes, dtype=object)
    if holes.ndim != 1:
        raise TypeError("holes is not a one-dimensional array.")

    indices = np.asarray(indices, dtype=np.int32)
    if indices.ndim != 1:
        raise TypeError("indices is not a one-dimensional array.")

    if holes.shape[0] != indices.shape[0]:
        raise ValueError("holes and indices do not have equal size.")

    if shells.shape[0] == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=object)

    if (indices >= shells.shape[0]).any():
        raise ValueError("Some indices are of bounds of the shells array.")  

    if np.any(indices[1:] < indices[:indices.shape[0] - 1]):
        raise ValueError("The indices should be sorted.")  

    cdef Py_ssize_t n_poly = shells.shape[0]
    cdef object[:] shells_view = shells
    cdef object[:] holes_view = holes
    cdef int[:] indices_view = indices

    # get the holes count per polygon (this raises on negative indices)
    cdef int[:] hole_count = np.bincount(indices, minlength=n_poly).astype(np.int32)

    # A temporary array for the holes that will be given to CreatePolygon
    # Its size equals max(hole_count) to accomodate the largest polygon.
    temp_holes = np.empty(shape=(np.max(hole_count), ), dtype=np.intp)
    cdef np.intp_t[:] temp_holes_view = temp_holes

    # The final target array
    result = np.empty(shape=(n_poly, ), dtype=object)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for poly_idx in range(n_poly):
            n_holes = 0

            # get the shell
            if PyGEOS_GetGEOSGeometry(<PyObject *>shells_view[poly_idx], &shell) == 0:
                raise TypeError(
                    "One of the arguments is of incorrect type. Please provide only Geometry objects."
                )

            # return None for missing shells (ignore possibly present holes)
            if shell == NULL:
                result_view[poly_idx] = PyGEOS_CreateGeometry(NULL, geos_handle)
                hole_idx_1 += hole_count[poly_idx]
                continue

            geom_type = GEOSGeomTypeId_r(geos_handle, shell)
            if geom_type == -1:
                return  # GEOSException is raised by get_geos_handle
            elif geom_type != 2:
                raise TypeError(
                    f"One of the shells has unexpected geometry type {geom_type}."
                )

            # fill the temporary array with holes belonging to this polygon
            for poly_hole_idx in range(hole_count[poly_idx]):
                if PyGEOS_GetGEOSGeometry(<PyObject *>holes_view[hole_idx_1 + poly_hole_idx], &hole) == 0:
                    _deallocate_arr(geos_handle, temp_holes_view, n_holes)
                    raise TypeError(
                        "One of the arguments is of incorrect type. Please provide only Geometry objects."
                    )

                # ignore missing holes
                if hole == NULL:
                    continue

                # check the type
                geom_type = GEOSGeomTypeId_r(geos_handle, hole)
                if geom_type == -1:
                    _deallocate_arr(geos_handle, temp_holes_view, n_holes)
                    return  # GEOSException is raised by get_geos_handle
                elif geom_type != 2:
                    _deallocate_arr(geos_handle, temp_holes_view, n_holes)
                    raise TypeError(
                        f"One of the holes has unexpected geometry type {geom_type}."
                    )

                # assign to the temporary geometry array  
                hole = GEOSGeom_clone_r(geos_handle, hole)
                if hole == NULL:
                    _deallocate_arr(geos_handle, temp_holes_view, n_holes)
                    return  # GEOSException is raised by get_geos_handle                 
                temp_holes_view[n_holes] = <np.intp_t>hole
                n_holes += 1

            # clone the shell as the polygon will take ownership
            shell = GEOSGeom_clone_r(geos_handle, shell)
            if shell == NULL:
                _deallocate_arr(geos_handle, temp_holes_view, n_holes)
                return  # GEOSException is raised by get_geos_handle

            # create the polygon
            poly = GEOSGeom_createPolygon_r(
                geos_handle,
                shell, 
                <GEOSGeometry**> &temp_holes_view[0],
                n_holes
            )
            if poly == NULL:
                # GEOSGeom_createPolygon_r should take ownership of the input geometries,
                # but it doesn't in case of an exception. We prefer a memory leak here over
                # a possible segfault in the future. The pre-emptive type check already covers
                # all known cases that GEOSGeom_createPolygon_r errors, so this should never
                # happen anyway. https://trac.osgeo.org/geos/ticket/1111.
                return  # GEOSException is raised by get_geos_handle

            result_view[poly_idx] = PyGEOS_CreateGeometry(poly, geos_handle)

            hole_idx_1 += hole_count[poly_idx]

    return result
