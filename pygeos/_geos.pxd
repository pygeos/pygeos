cdef extern from "geos_c.h":
    # GEOS C API types and functions.
    # Note: GEOS functions in Cython must be called using the get_geos_handle context
    # manager.

    ctypedef void *GEOSContextHandle_t

    GEOSContextHandle_t GEOS_init_r()
    void GEOS_finish_r(GEOSContextHandle_t handle)

    ctypedef struct GEOSGeometry

    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle,
                                           const GEOSGeometry* g,
                                           int n) except NULL
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle,
                                   const GEOSGeometry* g) except NULL


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef GEOSContextHandle_t __enter__(self)
