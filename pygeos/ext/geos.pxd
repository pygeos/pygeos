cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    GEOSContextHandle_t GEOS_init_r()
    ctypedef struct GEOSGeometry


cdef GEOSContextHandle_t get_geos_handle()