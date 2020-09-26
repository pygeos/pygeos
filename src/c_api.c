#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#define PyGEOS_API_Module

#include "c_api.h"
#include "geos.h"
#include "pygeom.h"

#include "stdio.h"

// TODO: provide typedefs as opaque pointers?

// TODO: provide context handle from here instead of via GEOS?
// use GEOS_INIT / GEOS_FINISH

extern char* PyGEOS_GEOS_API_Version(void) {
    return GEOS_CAPI_VERSION;
}

extern PyObject* PyGEOSCreateGeom(GEOSGeometry *ptr, GEOSContextHandle_t ctx) {
    return GeometryObject_FromGEOS(ptr, ctx);
}

extern char PyGEOSGetGEOSGeom(GeometryObject *obj, GEOSGeometry **out) {
    return get_geom(obj, out);
}

