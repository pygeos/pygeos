/* See
 * https://docs.python.org/3.6/extending/extending.html#providing-a-c-api-for-an-extension-module
 */

#ifndef _PYGEOS_API_H
#define _PYGEOS_API_H

#include <Python.h>

#include "geos.h"

/* C API functions */
#define PyGEOS_GetGeom_NUM 0
#define PyGEOS_GetGeom_RETURN char
#define PyGEOS_GetGeom_PROTO (PyObject * obj, GEOSGeometry * *out)

/* Total number of C API pointers */
#define PyGEOS_API_pointers 1

#ifdef PyGEOS_API_Module
/* This section is used when compiling c_api.c */

extern PyGEOS_GetGeom_RETURN PyGEOS_GetGeom PyGEOS_GetGeom_PROTO;

#else
/* This section is used in modules that use pygeos' C API */

static void** PyGEOS_API;

#define PyGEOS_GetGeom \
  (*(PyGEOS_GetGeom_RETURN(*) PyGEOS_GetGeom_PROTO)PyGEOS_API[PyGEOS_GetGeom_NUM])

/* Return -1 on error, 0 on success.
 * PyCapsule_Import will set an exception if there's an error.
 */
static int import_pygeos_api(void) {
  PyGEOS_API = (void**)PyCapsule_Import("pygeos.lib._C_API", 0);
  return (PyGEOS_API == NULL) ? -1 : 0;
}

#endif
#endif
