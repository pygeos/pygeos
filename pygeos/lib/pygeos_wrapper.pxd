"""
Provides a wrapper for the pygeos.lib C extension C API for use in Cython.
Internally, the pygeos C extension uses a PyCapsule to provide run-time access
to function pointers within the C API.

To use these functions, you must first call the following in each top-level
function in each Cython module:
`import_pygeos_core_api()`

If you get unexplained segfaults, this is a likely culprit.

This uses a macro to dynamically load the functions from pointers in the PyCapsule.

Each C function in pygeos.lib exposed in the C API must be specially-wrapped
to enable this capability.
"""

from cpython.ref cimport PyObject
cimport numpy as np

from pygeos.lib.geos_wrapper cimport *


cdef extern from "pygeom.h":
    ctypedef struct GeometryObject:
        np.intp_t ptr

    ctypedef class pygeos.lib.core.Geometry [object GeometryObject]:
        cdef np.intp_t _ptr "ptr"


cdef extern from "c_api.h":
    # pygeos.lib.core C API loader; returns -1 on error
    # MUST be called before calling other C API functions
    # returns -1 on failure
    int import_pygeos_core_api() except -1

    # C functions provided by the pygeos.lib C API
    char* PyGEOS_GEOS_API_Version()
    PyObject* PyGEOSCreateGeom(GEOSGeometry *ptr, GEOSContextHandle_t ctx)
    char PyGEOSGetGEOSGeom(GeometryObject *obj, GEOSGeometry **out)
