import cython

import numpy as np
cimport numpy as np


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    GEOSContextHandle_t GEOS_init_r()
    ctypedef struct GEOSGeometry
    int GEOSGetNumGeometries_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    int GEOSGetNumCoordinates_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    int GEOSGetNumInteriorRings_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    const GEOSGeometry *GEOSGetGeometryN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n)
    const GEOSGeometry *GEOSGetExteriorRing_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    const GEOSGeometry *GEOSGetInteriorRingN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n)


cdef extern from "pygeom.h":

    ctypedef struct GeometryObject:
        np.intp_t ptr

    ctypedef class pygeos.lib.Geometry [object GeometryObject]:
        cdef np.intp_t _ptr "ptr"


cdef GEOSContextHandle_t get_geos_handle():
    cdef GEOSContextHandle_t handle

    handle = GEOS_init_r()
    return handle


@cython.boundscheck(False)
@cython.wraparound(False)
def get_offset_array_sizes(array):
    """
    Get the size of the offset arrays for MultiPolygons
    (for second and third level)
    """
    cdef Py_ssize_t idx
    cdef unsigned int n = array.size
    cdef int n_geoms, idx_geom, n_rings, n_coords
    cdef int counter_geoms, counter_rings

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *new_geom
    cdef GeometryObject *pygeom
    cdef np.intp_t pgeom

    counter_geoms = 0
    counter_rings = 0

    geos_handle = get_geos_handle()

    for idx in xrange(n):
        pygeom = <GeometryObject *> array[idx]
        pgeom = pygeom.ptr
        geom = <GEOSGeometry *>pgeom
        n_geoms = GEOSGetNumGeometries_r(geos_handle, geom)
        counter_geoms += n_geoms

        for idx_geom in xrange(n_geoms):
            new_geom = GEOSGetGeometryN_r(geos_handle, geom, idx_geom)
            n_rings = GEOSGetNumInteriorRings_r(geos_handle, new_geom)
            counter_rings += n_rings + 1

    return counter_geoms, counter_rings


@cython.boundscheck(False)
@cython.wraparound(False)
def get_offset_arrays(array):
    """
    Get the offset arrays for an array of MultiPolygons.

    Currently assumes you have only (Multi)Polygons. Doesn't check yet for
    other geometry types / missing values.

    """
    cdef Py_ssize_t idx, idx_rings, idx_coords
    cdef unsigned int n = array.size
    cdef int n_geoms, n_rings, n_coords
    cdef int idx_geom, idx_ring
    cdef int counter_geoms, counter_rings, counter_coords
    cdef int[:] result_geoms, result_rings, result_coords

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSGeometry *geom
    cdef GEOSGeometry *new_geom
    cdef GEOSGeometry *ring
    cdef GeometryObject *pygeom
    cdef np.intp_t pgeom

    n_rings, n_coords = get_offset_array_sizes(array)

    result_geoms = np.empty(n + 1, dtype=np.intc)
    result_rings = np.empty(n_rings + 1, dtype=np.intc)
    result_coords = np.empty(n_coords + 1, dtype=np.intc)

    counter_geoms, counter_rings, counter_coords = 0, 0, 0
    result_geoms[0] = counter_geoms

    idx_rings = 0
    result_rings[idx_rings] = counter_rings

    idx_coords = 0
    result_coords[idx_coords] = counter_rings

    geos_handle = get_geos_handle()

    for idx in xrange(n):
        pygeom = <GeometryObject *> array[idx]
        pgeom = pygeom.ptr
        geom = <GEOSGeometry *>pgeom
        n_geoms = GEOSGetNumGeometries_r(geos_handle, geom)
        counter_geoms += n_geoms
        result_geoms[idx+1] = counter_geoms

        for idx_geom in xrange(n_geoms):
            new_geom = GEOSGetGeometryN_r(geos_handle, geom, idx_geom)
            n_rings = GEOSGetNumInteriorRings_r(geos_handle, new_geom)
            counter_rings += n_rings + 1
            idx_rings += 1
            result_rings[idx_rings] = counter_rings

            ring = GEOSGetExteriorRing_r(geos_handle, new_geom)
            n_coords = GEOSGetNumCoordinates_r(geos_handle, ring)
            counter_coords += n_coords
            idx_coords += 1
            result_coords[idx_coords] = counter_coords

            for idx_ring in xrange(n_rings):
                ring = GEOSGetInteriorRingN_r(geos_handle, new_geom, idx_ring)
                n_coords = GEOSGetNumCoordinates_r(geos_handle, ring)
                counter_coords += n_coords
                idx_coords += 1
                result_coords[idx_coords] = counter_coords

    return (result_geoms.base, result_rings.base, result_coords.base)
