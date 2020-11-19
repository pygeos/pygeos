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

    ctypedef struct GEOSCoordSequence

    GEOSCoordSequence* GEOSCoordSeq_create_r(GEOSContextHandle_t handle,
                                              unsigned int size,
                                              unsigned int dims) nogil except NULL

    int GEOSCoordSeq_setX_r(GEOSContextHandle_t handle,
                            GEOSCoordSequence* s, unsigned int idx,
                            double val) nogil except 0

    int GEOSCoordSeq_setY_r(GEOSContextHandle_t handle,
                            GEOSCoordSequence* s, unsigned int idx,
                            double val) nogil except 0

    void GEOSCoordSeq_destroy_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil

    ctypedef struct GEOSGeometry

    GEOSGeometry* GEOSGeom_createLinearRing_r(GEOSContextHandle_t handle,
                                          GEOSCoordSequence* cs) nogil except NULL

    GEOSGeometry* GEOSGeom_createPolygon_r(GEOSContextHandle_t handle,
                                          GEOSGeometry* shell,
                                          GEOSGeometry** holes,
                                          unsigned int nholes) nogil except NULL

    int GEOSGeomTypeId_r(GEOSContextHandle_t handle,
                         const GEOSGeometry* g) nogil except -1

    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle,
                                           const GEOSGeometry* g,
                                           int n) nogil except NULL

    int GEOSGeom_getDimensions_r(GEOSContextHandle_t handle,
                                 const GEOSGeometry* g) nogil

    GEOSGeometry* GEOSIntersection_r(GEOSContextHandle_t handle,
                                     const GEOSGeometry* g1,
                                     const GEOSGeometry* g2) nogil except NULL

    int GEOSGetNumCoordinates_r(GEOSContextHandle_t handle,
                               const GEOSGeometry* g) nogil except -1

    int GEOSGetNumGeometries_r(GEOSContextHandle_t handle,
                               const GEOSGeometry* g) nogil except -1

    char GEOSisEmpty_r(GEOSContextHandle_t handle, const GEOSGeometry* g) nogil

    GEOSGeometry* GEOSSimplify_r(GEOSContextHandle_t handle,
                                 const GEOSGeometry* g,
                                 double tolerance) nogil except NULL

    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle,
                                   const GEOSGeometry* g) nogil except NULL

    void GEOSGeom_destroy_r(GEOSContextHandle_t handle, GEOSGeometry* a) nogil

    int GEOSGeom_getXMax_r(GEOSContextHandle_t handle, const GEOSGeometry* g,
                           double* value) nogil except 0

    int GEOSGeom_getYMax_r(GEOSContextHandle_t handle, const GEOSGeometry* g,
                           double* value) nogil except 0

    int GEOSGeom_getXMin_r(GEOSContextHandle_t handle, const GEOSGeometry* g,
                           double* value) nogil except 0

    int GEOSGeom_getYMin_r(GEOSContextHandle_t handle, const GEOSGeometry* g,
                           double* value) nogil except 0


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef GEOSContextHandle_t __enter__(self)
