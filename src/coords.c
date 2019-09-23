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

static char get_coordinates(GEOSContextHandle_t, GEOSGeometry *, PyArrayObject *, npy_intp *);

static char get_coordinates_simple(GEOSContextHandle_t context, GEOSGeometry *geom,
                                   PyArrayObject *out, npy_intp *cursor) {
    unsigned int n, i;
    double *x, *y;
    const GEOSCoordSequence *seq = GEOSGeom_getCoordSeq_r(context, geom);
    if (seq == NULL) { return 0; }
    if (GEOSCoordSeq_getSize_r(context, seq, &n) == 0) { return 0; }
    for(i = 0; i < n; i++, *cursor += 1) {
        x = PyArray_GETPTR2(out, 0, *cursor);
        y = PyArray_GETPTR2(out, 1, *cursor);
        if (GEOSCoordSeq_getX_r(context, seq, i, x) == 0) { return 0; }
        if (GEOSCoordSeq_getY_r(context, seq, i, y) == 0) { return 0; }
    }
    return 1;
}

static char get_coordinates_polygon(GEOSContextHandle_t context, GEOSGeometry *geom,
                                    PyArrayObject *out, npy_intp *cursor) {
    int n, i;
    GEOSGeometry *ring;

    ring = (GEOSGeometry *) GEOSGetExteriorRing_r(context, geom);
    if (ring == NULL) { return 0; }
    if (!get_coordinates_simple(context, ring, out, cursor)) { return 0; }

    n = GEOSGetNumInteriorRings_r(context, geom);
    if (n == -1) { return 0; }
    for(i = 0; i < n; i++) {
        ring = (GEOSGeometry *) GEOSGetInteriorRingN_r(context, geom, i);
        if (ring == NULL) { return 0; }
        if (!get_coordinates_simple(context, ring, out, cursor)) { return 0; }
    }
    return 1;
}

static char get_coordinates_collection(GEOSContextHandle_t context, GEOSGeometry *geom,
                                       PyArrayObject *out, npy_intp *cursor) {
    int n, i;
    GEOSGeometry *sub_geom;

    n = GEOSGetNumGeometries_r(context, geom);
    if (n == -1) { return 0; }
    for(i = 0; i < n; i++) {
        sub_geom = (GEOSGeometry *) GEOSGetGeometryN_r(context, geom, i);
        if (sub_geom == NULL) { return 0; }
        if (!get_coordinates(context, sub_geom, out, cursor)) { return 0; }
    }
    return 1;
}

static char get_coordinates(GEOSContextHandle_t context, GEOSGeometry *geom,
                            PyArrayObject *out, npy_intp *cursor) {
    int type = GEOSGeomTypeId_r(context, geom);
    if ((type == 0) | (type == 1) | (type == 2)) {
        return get_coordinates_simple(context, geom, out, cursor);
    } else if (type == 3) {
        return get_coordinates_polygon(context, geom, out, cursor);
    } else if ((type >= 4) & (type <= 7)) {
        return get_coordinates_collection(context, geom, out, cursor);
    } else {
        return 0;
    }
}


static void *set_coordinates_simple(GEOSContextHandle_t context, GEOSGeometry *geom,
                                   int type, PyArrayObject *coords, npy_intp *cursor) {
    unsigned int n, i, dims;
    double *x, *y;
    GEOSGeometry *ret;

    /* Investigate the current (const) CoordSequence */
    const GEOSCoordSequence *seq = GEOSGeom_getCoordSeq_r(context, geom);
    if (seq == NULL) { return NULL; }
    if (GEOSCoordSeq_getSize_r(context, seq, &n) == 0) { return NULL; }
    if (GEOSCoordSeq_getDimensions_r(context, seq, &dims) == 0) { return NULL; }

    /* Create a new one to fill with the new coordinates */
    GEOSCoordSequence *seq_new = GEOSCoordSeq_create_r(context, n, dims);
    if (seq_new == NULL) { return NULL; }

    for(i = 0; i < n; i++, *cursor += 1) {
        x = PyArray_GETPTR2(coords, 0, *cursor);
        y = PyArray_GETPTR2(coords, 1, *cursor);
        if (GEOSCoordSeq_setX_r(context, seq_new, i, *x) == 0) { goto fail; }
        if (GEOSCoordSeq_setY_r(context, seq_new, i, *y) == 0) { goto fail; }
    }

    /* Construct a new geometry */
    if (type == 0) {
        ret = GEOSGeom_createPoint_r(context, seq_new);
    } else if (type == 1) {
        ret = GEOSGeom_createLineString_r(context, seq_new);
    } else if (type == 2) {
        ret = GEOSGeom_createLinearRing_r(context, seq_new);
    } else {
        goto fail;
    }
    /* Do not destroy the seq_new if ret is NULL; will lead to segfaults */
    return ret;

    fail:
        GEOSCoordSeq_destroy_r(context, seq_new);
        return NULL;
}


static void *set_coordinates_polygon(GEOSContextHandle_t context, GEOSGeometry *geom,
                                     PyArrayObject *coords, npy_intp *cursor) {
    int i, dims;
    GEOSGeometry *ring, *hole;

    int n = GEOSGetNumInteriorRings_r(context, geom);
    if (n == -1) { return NULL; }
    GEOSGeometry **holes = malloc(sizeof(void *) * n);

    /* create the exterior ring */
    ring = (GEOSGeometry *) GEOSGetExteriorRing_r(context, geom);
    if (ring == NULL) { return NULL; }
    ring = set_coordinates_simple(context, ring, 2, out, cursor)
    if (ring == NULL) ( return NULL; )

    for(i = 0; i < n; i++) {
        hole = (GEOSGeometry *) GEOSGetInteriorRingN_r(context, geom, i);
        if (hole == NULL) { return NULL; }
        hole = set_coordinates_simple(context, hole, 2, out, cursor)
        if (hole == NULL) { return NULL; }
    }

    return 1;



    /* Investigate the current (const) CoordSequence */
    const GEOSCoordSequence *seq = GEOSGeom_getCoordSeq_r(context, geom);
    if (seq == NULL) { return NULL; }
    if (GEOSCoordSeq_getSize_r(context, seq, &n) == 0) { return NULL; }
    if (GEOSCoordSeq_getDimensions_r(context, seq, &dims) == 0) { return NULL; }

    /* Create a new one to fill with the new coordinates */
    GEOSCoordSequence *seq_new = GEOSCoordSeq_create_r(context, n, dims);
    if (seq_new == NULL) { return NULL; }

    for(i = 0; i < n; i++, *cursor += 1) {
        x = PyArray_GETPTR2(coords, 0, *cursor);
        y = PyArray_GETPTR2(coords, 1, *cursor);
        if (GEOSCoordSeq_setX_r(context, seq_new, i, *x) == 0) { goto fail; }
        if (GEOSCoordSeq_setY_r(context, seq_new, i, *y) == 0) { goto fail; }
    }

    /* Construct a new geometry */
    if (type == 0) {
        ret = GEOSGeom_createPoint_r(context, seq_new);
    } else if (type == 1) {
        ret = GEOSGeom_createLineString_r(context, seq_new);
    } else if (type == 2) {
        ret = GEOSGeom_createLinearRing_r(context, seq_new);
    } else {
        goto fail;
    }
    /* Do not destroy the seq_new if ret is NULL; will lead to segfaults */
    return ret;

    fail:
        GEOSCoordSeq_destroy_r(context, seq_new);
        return NULL;
}

static void *set_coordinates(GEOSContextHandle_t context, GEOSGeometry *geom,
                            PyArrayObject *coords, npy_intp *cursor) {
    int type = GEOSGeomTypeId_r(context, geom);
    if ((type == 0) | (type == 1) | (type == 2)) {
        return set_coordinates_simple(context, geom, type, coords, cursor);
    } else if (type == 3) {
        return set_coordinates_polygon(context, geom, coords, cursor);
    } else if ((type >= 4) & (type <= 7)) {
        return set_coordinates_collection(context, geom, type, coords, cursor);
    } else {
        return NULL;
    }
}

npy_intp CountCoords(PyArrayObject *arr)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    int ret;
    npy_intp result = 0;
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

    do {
        obj = *(GeometryObject **) dataptr[0];
        if (!PyObject_IsInstance((PyObject *) obj, (PyObject *) &GeometryType)) { continue; }
        geom = obj->ptr;
        if (geom == NULL) { continue; }
        ret = GEOSGetNumCoordinates_r(context, geom);
        if (ret < 0) {
            NpyIter_Deallocate(iter);
            GEOS_finish_r(context);
            return -1;
        }
        result += ret;
    } while(iternext(iter));

    NpyIter_Deallocate(iter);
    GEOS_finish_r(context);

    return result;
}


PyObject *GetCoords(PyArrayObject *arr)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp cursor;
    GeometryObject *obj;
    GEOSGeometry *geom;
    GEOSContextHandle_t context;

    /* create a coordinate array with the appropriate dimensions */
    npy_intp size = CountCoords(arr);
    if (size == -1) {
        return Py_None;
    }
    npy_intp dims[2] = {2, size};
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (result == NULL) {
        return Py_None;
    }

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(arr) == 0) {
        return (PyObject *) result;
    }

    iter = NpyIter_New(arr, NPY_ITER_READONLY|NPY_ITER_REFS_OK,
                       NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        Py_DECREF(result);
        return Py_None;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        Py_DECREF(result);
        NpyIter_Deallocate(iter);
        return Py_None;
    }

    context = GEOS_init_r();
    if (context == NULL) {
        Py_DECREF(result);
        NpyIter_Deallocate(iter);
        return Py_None;
    }

    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);

    cursor = 0;
    do {
        obj = *(GeometryObject **) dataptr[0];
        if (!PyObject_IsInstance((PyObject *) obj, (PyObject *) &GeometryType)) { continue; }
        geom = obj->ptr;
        if (geom == NULL) { continue; }
        if (!get_coordinates(context, geom, result, &cursor)) {
            Py_DECREF(result);
            NpyIter_Deallocate(iter);
            GEOS_finish_r(context);
            return Py_None;
        }
    } while(iternext(iter));

    NpyIter_Deallocate(iter);
    GEOS_finish_r(context);

    return (PyObject *) result;
}


PyObject *SetCoords(PyArrayObject *geoms, PyArrayObject *coords)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp cursor;
    GeometryObject *obj;
    PyObject *new_obj;
    PyObject **obj_ptr;
    GEOSGeometry *geom, *new_geom;
    GEOSContextHandle_t context;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(geoms) == 0) {
        Py_INCREF((PyObject *) geoms);
        return (PyObject *) geoms;
    }

    iter = NpyIter_New(geoms, NPY_ITER_READWRITE|NPY_ITER_REFS_OK,
                       NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        return Py_None;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return Py_None;
    }

    context = GEOS_init_r();
    if (context == NULL) {
        NpyIter_Deallocate(iter);
        return Py_None;
    }

    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);

    cursor = 0;

    do {
        obj_ptr = (PyObject **)dataptr[0];
        obj = *(GeometryObject **)obj_ptr;
        if (!PyObject_IsInstance((PyObject *) obj, (PyObject *) &GeometryType)) { continue; }
        geom = obj->ptr;
        if (geom == NULL) { continue; }
        new_geom = set_coordinates(context, geom, coords, &cursor);
        if (new_geom == NULL) {
            NpyIter_Deallocate(iter);
            GEOS_finish_r(context);
            return Py_None;
        }
        new_obj = GeometryObject_FromGEOS(&GeometryType, new_geom);
        Py_XDECREF(obj);
        *obj_ptr = new_obj;
    } while(iternext(iter));

    NpyIter_Deallocate(iter);
    GEOS_finish_r(context);

    Py_INCREF((PyObject *) geoms);
    return (PyObject *) geoms;
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


PyObject *PyGetCoords(PyObject *self, PyObject *args)
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
    return GetCoords((PyArrayObject *) arr);
}

PyObject *PySetCoords(PyObject *self, PyObject *args)
{
    PyObject *geoms;
    PyObject *coords;

    if (!PyArg_ParseTuple(args, "OO", &geoms, &coords)) {
        return NULL;
    }
    if ((!PyArray_Check(geoms)) | (!PyArray_Check(coords))) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }
    if (!PyArray_ISOBJECT((PyArrayObject *) geoms)) {
        PyErr_SetString(PyExc_TypeError, "Geometry array should be of object dtype");
        return NULL;
    }
    if ((PyArray_TYPE((PyArrayObject *) coords)) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Coordinate array should be of float64 dtype");
        return NULL;
    }
    geoms = SetCoords((PyArrayObject *) geoms, (PyArrayObject *) coords);
    if (geoms == Py_None) {
        return NULL;
    }
    return geoms;
}
