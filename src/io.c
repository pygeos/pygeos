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


PyObject *FromShapely(PyArrayObject *arr)
{
    NpyIter *iter1, *iter2;
    NpyIter_IterNextFunc *iternext1, *iternext2;
    char** dataptr1;
    char** dataptr2;
    PyObject *obj, *attr;
    size_t obj_ptr;
    PyObject **ret_ptr;
    GEOSGeometry *geom;
    GEOSGeometry *geom_copy;
    PyObject *new_obj;
    GEOSContextHandle_t context;

    /* create an object array of same size as output array */
    PyArrayObject *result = (PyArrayObject *) PyArray_NewLikeArray(arr, NPY_ANYORDER, NULL, 0);
    if (result == NULL) {
        return Py_None;
    }

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(arr) == 0) {
        return (PyObject *) result;
    }

    iter1 = NpyIter_New(arr, NPY_ITER_READONLY|NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter1 == NULL) {
        Py_DECREF(result);
        return Py_None;
    }
    iter2 = NpyIter_New(result, NPY_ITER_READONLY|NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter2 == NULL) {
        Py_DECREF(result);
        return Py_None;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext1 = NpyIter_GetIterNext(iter1, NULL);
    if (iternext1 == NULL) {
        Py_DECREF(result);
        NpyIter_Deallocate(iter1);
        return Py_None;
    }
    iternext2 = NpyIter_GetIterNext(iter2, NULL);
    if (iternext2 == NULL) {
        Py_DECREF(result);
        NpyIter_Deallocate(iter1);
        NpyIter_Deallocate(iter2);
        return Py_None;
    }

    context = GEOS_init_r();
    if (context == NULL) {
        Py_DECREF(result);
        NpyIter_Deallocate(iter1);
        NpyIter_Deallocate(iter2);
        return Py_None;
    }

    /* The location of the data pointer which the iterator may update */
    dataptr1 = NpyIter_GetDataPtrArray(iter1);
    dataptr2 = NpyIter_GetDataPtrArray(iter2);

    do {
        obj = *(PyObject **) dataptr1[0];
        ret_ptr = (PyObject **) dataptr2[0];

        if (obj == Py_None) {
            *ret_ptr = NULL;
        } else {
            attr = PyObject_GetAttrString(obj, "__geom__");
            if (attr == Py_None){ 
                goto fail;
            }
            obj_ptr = PyLong_AsSize_t(attr);
            geom = (uintptr_t) obj_ptr;
            geom_copy = GEOSGeom_clone_r(context, geom);
            if (geom_copy == NULL) { goto fail; }

            new_obj = GeometryObject_FromGEOS(&GeometryType, geom_copy);
            Py_XDECREF(obj);
            *ret_ptr = new_obj;
        }
    } while(iternext1(iter1) && iternext2(iter2));

    NpyIter_Deallocate(iter1);
    NpyIter_Deallocate(iter2);
    GEOS_finish_r(context);

    return (PyObject *) result;

    fail:
        Py_DECREF(result);
        NpyIter_Deallocate(iter1);
        NpyIter_Deallocate(iter2);
        GEOS_finish_r(context);
        return Py_None;
}


PyObject *PyFromShapely(PyObject *self, PyObject *args)
{
    PyObject *arr;

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
    return FromShapely((PyArrayObject *) arr);
}
