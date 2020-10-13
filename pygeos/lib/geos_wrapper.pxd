cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    GEOSContextHandle_t GEOS_init_r()
    ctypedef struct GEOSGeometry

    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n)
    int GEOSGetNumGeometries_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    int GEOSisEmpty_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle, const GEOSGeometry* g)


cdef GEOSContextHandle_t get_geos_handle()
