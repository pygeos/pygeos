from pygeos._geos cimport GEOSContextHandle_t, GEOSGeometry


cdef int _bounds(
    GEOSContextHandle_t geos_handle,
    const GEOSGeometry *geom,
    double *xmin,
    double *ymin,
    double *xmax,
    double *ymax
) nogil except 0
