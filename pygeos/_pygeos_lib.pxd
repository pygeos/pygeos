cimport numpy as np
from ._geos cimport GEOSGeometry

cdef extern from "pygeom.h":
    ctypedef struct GeometryObject:
        np.intp_t ptr

    ctypedef class pygeos.lib.Geometry [object GeometryObject]:
        cdef np.intp_t _ptr "ptr"

    char get_geom(GeometryObject* obj, GEOSGeometry** out)
