"""
Provides a wrapper for GEOS types and functions.

Note: GEOS functions in Cython must be called using the get_geos_handle context manager.
Example:
    with get_geos_handle() as geos_handle:
        SomeGEOSFunc(geos_handle, ...<other params>)
"""

cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t

    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil

    ctypedef struct GEOSGeometry

    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle,
                                           const GEOSGeometry* g,
                                           int n) nogil except NULL
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle,
                                   const GEOSGeometry* g) nogil except NULL

    int GEOSGetNumGeometries_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    int GEOSGetNumCoordinates_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    int GEOSGetNumInteriorRings_r(GEOSContextHandle_t handle, const GEOSGeometry* g)

    const GEOSGeometry* GEOSGetExteriorRing_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    const GEOSGeometry* GEOSGetInteriorRingN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n)


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef GEOSContextHandle_t __enter__(self)
