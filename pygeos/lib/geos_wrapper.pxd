cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    GEOSContextHandle_t GEOS_init_r()
    ctypedef struct GEOSGeometry

    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n) except NULL
    int GEOSGetNumGeometries_r(GEOSContextHandle_t handle, const GEOSGeometry* g) except -1
    int GEOSisEmpty_r(GEOSContextHandle_t handle, const GEOSGeometry* g) except 2
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle, const GEOSGeometry* g) except NULL
