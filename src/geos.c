#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "geos.h"

/* This initializes a global GEOS Context */
void *geos_context[1] = {NULL};
PyObject *geos_exception[1] = {NULL};

static void HandleGEOSError(const char *message, void *userdata) {
    PyErr_SetString(userdata, message);
}

static void HandleGEOSNotice(const char *message, void *userdata) {
    PyErr_WarnEx(PyExc_Warning, message, 1);
}

void geos_error_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}

void geos_notice_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}


int init_geos(PyObject *m)
{
    void *context_handle = GEOS_init_r();
    geos_exception[0] = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
    PyModule_AddObject(m, "GEOSException", geos_exception[0]);
    GEOSContext_setErrorMessageHandler_r(context_handle, HandleGEOSError, geos_exception[0]);
    GEOSContext_setNoticeMessageHandler_r(context_handle, HandleGEOSNotice, NULL);
    geos_context[0] = context_handle;  /* for global access */
    return 0;
}
