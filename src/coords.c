#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <math.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API
#include "numpy/arrayobject.h"

#include "numpy/ndarraytypes.h"
#include "numpy/npy_3kcompat.h"

#include "geos.h"
#include "pygeom.h"


npy_intp CountCoords(PyArrayObject* arr)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp coord_count;
    int ret;
    GeometryObject *obj;
    GEOSGeometry *geom;
    GEOSContextHandle_t context;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(arr) == 0) {
        return 0;
    }

    iter = NpyIter_New(arr, NPY_ITER_READONLY|NPY_ITER_REFS_OK,
                       NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        return -1;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    context = GEOS_init_r();
    if (context == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);

    coord_count = 0;
    do {
        obj = *(GeometryObject **) dataptr[0];
        if (!PyObject_IsInstance((PyObject *) obj, (PyObject *) &GeometryType)) { continue; }
        geom = obj->ptr;
        if (geom == NULL) { continue; }
        ret = GEOSGetNumCoordinates_r(context, geom);
        if (ret == -1) { continue; }
        coord_count += ret;
    } while(iternext(iter));

    NpyIter_Deallocate(iter);
    GEOS_finish_r(context);

    return coord_count;
}

PyObject *PyCountCoords(PyObject *self, PyObject *args)
{
    PyObject *arr;
    npy_intp ret;

    if (!PyArg_ParseTuple(args, "O", &arr)) {
        return NULL;
    }
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }
    if (!PyArray_ISOBJECT((PyArrayObject *) arr)) {
        PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
        return NULL;
    }
    ret = CountCoords((PyArrayObject *) arr);
    return PyLong_FromSsize_t(ret);
}
