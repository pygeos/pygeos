"""
Provides a wrapper for GEOS types and functions.

Note: GEOS functions in Cython must be called using the get_geos_handle context manager.
Example:
    with get_geos_handle() as geos_handle:
        SomeGEOSFunc(geos_handle, ...<other params>)
"""

cdef extern from "geos_c.h":
    # Types
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSCoordSequence

    # GEOS Context
    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil

    # Geometry functions
    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n) nogil except NULL
    int GEOSGeomTypeId_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil except -1

    # Predicates
    char GEOSisEmpty_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil except 2

    # Geometry creation / destruction
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle, const GEOSGeometry* g) nogil except NULL
    GEOSGeometry* GEOSGeom_createEmptyPoint_r(GEOSContextHandle_t handle) nogil except NULL
    GEOSGeometry* GEOSGeom_createPoint_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil except NULL
    GEOSGeometry* GEOSGeom_createLineString_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil except NULL
    GEOSGeometry* GEOSGeom_createLinearRing_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil except NULL
    GEOSGeometry* GEOSGeom_createCollection_r(GEOSContextHandle_t handle, int type, GEOSGeometry** geoms, unsigned int ngeoms) nogil except NULL
    void GEOSGeom_destroy_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil

    # Coordinate sequences
    const GEOSCoordSequence* GEOSGeom_getCoordSeq_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil except NULL
    GEOSCoordSequence* GEOSCoordSeq_create_r(GEOSContextHandle_t handle, unsigned int size, unsigned int dims) nogil except NULL
    void GEOSCoordSeq_destroy_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    int GEOSCoordSeq_getSize_r(GEOSContextHandle_t handle, const GEOSCoordSequence* s, unsigned int* size) nogil except 0
    int GEOSCoordSeq_getDimensions_r(GEOSContextHandle_t handle, const GEOSCoordSequence* s, unsigned int* ndim) nogil except 0
    int GEOSCoordSeq_getX_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double* val) nogil except 0
    int GEOSCoordSeq_getY_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double* val) nogil except 0
    int GEOSCoordSeq_getZ_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double* val) nogil except 0
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil except 0
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil except 0
    int GEOSCoordSeq_setZ_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil except 0

    GEOSGeometry* GEOSGeom_createCollection_r(
        GEOSContextHandle_t handle,
        int type,
        GEOSGeometry** geoms,
        unsigned int ngeoms
    ) nogil except NULL


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef GEOSContextHandle_t __enter__(self)
