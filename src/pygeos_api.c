#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#define PyGEOS_API_Module

#include "geos.h"
#include "pygeom.h"
#include "pygeos_api.h"

extern char* PyGEOS_GEOSVersion(void) { return GEOS_CAPI_VERSION; }

extern PyObject* PyGEOS_CreateGeom(GEOSGeometry* ptr, GEOSContextHandle_t ctx) {
  return GeometryObject_FromGEOS(ptr, ctx);
}

extern char PyGEOS_GetGeom(GeometryObject* obj, GEOSGeometry** out) {
  return get_geom(obj, out);
}
