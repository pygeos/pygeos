#ifndef _PYGEOS_API_H
#define _PYGEOS_API_H

// TODO: this might not be correct here, it isn't in example
#include <Python.h>
#include "geos.h"
#include "pygeom.h"

/* C API functions

Each function must provide 3 defines:
NUM: the index in function pointer array
RETURN: the return type
PROTO: function prototype

Important: each function must provide 2 sets of defines below and
provide an entry into PyGEOS_API in c_api.c module declaration block
 */

/* char* PyGEOS_GEOS_API_Version(void) - used to test that pygeos C API
 * functions properly */
#define PyGEOS_GEOS_API_Version_NUM 0
#define PyGEOS_GEOS_API_Version_RETURN char *
#define PyGEOS_GEOS_API_Version_PROTO (void)

/* PyObject* PyGEOS_CreateGeometry(GEOSGeometry *ptr, GEOSContextHandle_t ctx) */
#define PyGEOS_CreateGeometry_NUM 1
#define PyGEOS_CreateGeometry_RETURN PyObject *
#define PyGEOS_CreateGeometry_PROTO (GEOSGeometry * ptr, GEOSContextHandle_t ctx)

/* char PyGEOS_GetGEOSGeometry(GeometryObject *obj, GEOSGeometry **out) */
#define PyGEOS_GetGEOSGeometry_NUM 2
#define PyGEOS_GetGEOSGeometry_RETURN char
#define PyGEOS_GetGEOSGeometry_PROTO (GeometryObject * obj, GEOSGeometry * *out)

/* Total number of C API pointers */
#define PyGEOS_API_num_pointers 4

#ifdef PyGEOS_API_Module
/* This section is used when compiling pygeos.lib C extension.
 * Each API function needs to provide a corresponding *_PROTO here.
 */

extern PyGEOS_GEOS_API_Version_RETURN PyGEOS_GEOS_API_Version PyGEOS_GEOS_API_Version_PROTO;
extern PyGEOS_CreateGeometry_RETURN PyGEOS_CreateGeometry PyGEOS_CreateGeometry_PROTO;
extern PyGEOS_GetGEOSGeometry_RETURN PyGEOS_GetGEOSGeometry PyGEOS_GetGEOSGeometry_PROTO;

#else
/* This section is used in modules that use the pygeos C API
 * EAch API function needs to provide the lookup into PyGEOS_API as a
 * define statement.
*/

static void **PyGEOS_API;

#define PyGEOS_GEOS_API_Version \
    (*(PyGEOS_GEOS_API_Version_RETURN(*) PyGEOS_GEOS_API_Version_PROTO)PyGEOS_API[PyGEOS_GEOS_API_Version_NUM])

#define PyGEOS_CreateGeometry \
    (*(PyGEOS_CreateGeometry_RETURN(*) PyGEOS_CreateGeometry_PROTO)PyGEOS_API[PyGEOS_CreateGeometry_NUM])

#define PyGEOS_GetGEOSGeometry \
    (*(PyGEOS_GetGEOSGeometry_RETURN(*) PyGEOS_GetGEOSGeometry_PROTO)PyGEOS_API[PyGEOS_GetGEOSGeometry_NUM])

/* Dynamically load C API from PyCapsule.
 * It is necessary to call this prior to using C API functions in other modules.
 *
 * Returns 0 on success, -1 if error.
 * PyCapsule_Import will set an exception on error.
 */
static int
import_pygeos_core_api(void)
{
    PyGEOS_API = (void **)PyCapsule_Import("pygeos.lib.core._C_API", 0);
    return (PyGEOS_API == NULL) ? -1 : 0;
}

#endif

#endif /* !defined(_PYGEOS_API_H) */
