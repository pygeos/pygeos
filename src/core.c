#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#define PyGEOS_API_Module

#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL pygeos_UFUNC_API
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/ufuncobject.h>

#include "c_api.h"
#include "coords.h"
#include "geos.h"
#include "pygeom.h"
#include "strtree.h"
#include "ufuncs.h"

/* This tells Python what methods this module has. */
static PyMethodDef GeosModule[] = {
    {"count_coordinates", PyCountCoords, METH_VARARGS,
     "Counts the total amount of coordinates in a array with geometry objects"},
    {"get_coordinates", PyGetCoords, METH_VARARGS,
     "Gets the coordinates as an (N, 2) shaped ndarray of floats"},
    {"set_coordinates", PySetCoords, METH_VARARGS,
     "Sets coordinates to a geometry array"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "core", NULL, -1, GeosModule, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_core(void) {
  PyObject *m, *d;

  static void* PyGEOS_API[PyGEOS_API_num_pointers];
  PyObject* c_api_object;

  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  if (init_geos(m) < 0) {
    return NULL;
  };

  if (init_geom_type(m) < 0) {
    return NULL;
  };

  if (init_strtree_type(m) < 0) {
    return NULL;
  };

  d = PyModule_GetDict(m);

  import_array();
  import_umath();

  /* export the GEOS versions as python tuple and string */
  PyModule_AddObject(m, "geos_version",
                     PyTuple_Pack(3, PyLong_FromLong((long)GEOS_VERSION_MAJOR),
                                  PyLong_FromLong((long)GEOS_VERSION_MINOR),
                                  PyLong_FromLong((long)GEOS_VERSION_PATCH)));
  PyModule_AddObject(m, "geos_capi_version",
                     PyTuple_Pack(3, PyLong_FromLong((long)GEOS_CAPI_VERSION_MAJOR),
                                  PyLong_FromLong((long)GEOS_CAPI_VERSION_MINOR),
                                  PyLong_FromLong((long)GEOS_CAPI_VERSION_PATCH)));

  PyModule_AddObject(m, "geos_version_string", PyUnicode_FromString(GEOS_VERSION));
  PyModule_AddObject(m, "geos_capi_version_string",
                     PyUnicode_FromString(GEOS_CAPI_VERSION));

  if (init_ufuncs(m, d) < 0) {
    return NULL;
  };

  /* Initialize the C API pointer array */
  PyGEOS_API[PyGEOS_CreateGeometry_NUM] = (void*)PyGEOS_CreateGeometry;
  PyGEOS_API[PyGEOS_GetGEOSGeometry_NUM] = (void*)PyGEOS_GetGEOSGeometry;

  /* Create a Capsule containing the API pointer array's address */
  c_api_object = PyCapsule_New((void*)PyGEOS_API, "pygeos.lib.core._C_API", NULL);
  if (c_api_object != NULL) {
    PyModule_AddObject(m, "_C_API", c_api_object);
  }

  return m;
}