cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    GEOSContextHandle_t GEOS_init_r()
    void GEOS_finish_r(GEOSContextHandle_t handle)
    ctypedef struct GEOSGeometry

    int GEOSGetNumGeometries_r(GEOSContextHandle_t handle, const GEOSGeometry* g) except -1
