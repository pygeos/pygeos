cimport numpy as np
from cpython cimport PyObject
from ._geos cimport GEOSGeometry, GEOSContextHandle_t


cdef extern from "pygeos_api.h":
    # pygeos.lib C API loader; returns -1 on error
    # MUST be called before calling other C API functions
    int import_pygeos_api() except -1

    # C functions provided by the pygeos.lib C API
    char PyGEOS_GetGeom(PyObject *obj, GEOSGeometry **out)
