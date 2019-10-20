#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "strtree.h"
#include "geos.h"
#include "pygeom.h"
#include "kvec.h"


static PyObject *STRtree_new(PyTypeObject *type, PyObject *args,
                             PyObject *kwds)
{
    int node_capacity;
    PyObject *arr;
    void *tree, *ptr;
    npy_intp n, i;
    GEOSGeometry *geom;
    GeometryObject *obj;
    GEOSContextHandle_t context = geos_context[0];

    if (!PyArg_ParseTuple(args, "Oi", &arr, &node_capacity)) {
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
    if (PyArray_NDIM((PyArrayObject *) arr) != 1) {
        PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
        return NULL;
    }

    tree = GEOSSTRtree_create_r(context, (size_t) node_capacity);
    if (tree == NULL) {
        return NULL;
    }

    n = PyArray_SIZE((PyArrayObject *) arr);
    for(i = 0; i < n; i++) {
        /* get the geometry */
        ptr = PyArray_GETPTR1((PyArrayObject *) arr, i);
        obj = *(GeometryObject **) ptr;
        /* skip incase obj was no geometry or None */
        if (!get_geom(obj, &geom)) { continue; }
        if (geom == NULL) { continue; }
        /* perform the insert */
        Py_INCREF(obj);   /* STRtree holds a reference to each GeometryObject */
        GEOSSTRtree_insert_r(context, tree, geom, (void *) obj );
    }

    STRtree *self = (STRtree *) type->tp_alloc(type, 0);
    if (self == NULL) {
        GEOSSTRtree_destroy_r(context, tree);
        return NULL;
    }
    self->ptr = tree;
    return (PyObject *) self;
}

void strtree_dealloc_callback(void *item, void *vec)
{
    Py_XDECREF((PyObject *) item);
}

static void STRtree_dealloc(STRtree *self)
{
    void *context = geos_context[0];
    if (self->ptr != NULL) {
        /* Decrease the refcount of each GeometryObject in the STRTree */
        GEOSSTRtree_iterate_r(context, self->ptr, strtree_dealloc_callback, NULL);
        GEOSSTRtree_destroy_r(context, self->ptr);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

/* Callback to give to strtree_query
 * Given the value returned from each intersecting geometry it inserts that
 * value (typically an index) into the given size_vector */

void strtree_query_callback(void *item, void *user_data)
{
    kv_push(PyObject *, *(geom_array *)user_data, (PyObject *) item);
}

static PyObject *STRtree_query(STRtree *self, PyObject *envelope) {
    GEOSContextHandle_t context = geos_context[0];
    GEOSGeometry *geom;
    geom_array arr; // Resizable array for matches for each geometry
    npy_intp i, size;
    PyObject *obj;
    PyObject **ptr;

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }
    if (!get_geom((GeometryObject *) envelope, &geom)) {
        PyErr_SetString(PyExc_TypeError, "Invalid geometry");
        return NULL;
    }
    if (geom == NULL) {
        PyErr_SetString(PyExc_TypeError, "Missing geometry");
        return NULL;
    }

    kv_init(arr);
    GEOSSTRtree_query_r(context, self->ptr, geom, strtree_query_callback, &arr);

    /* create a geometry array with the appropriate dimensions */
    size = kv_size(arr);
    npy_intp dims[1] = {size};
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_OBJECT);
    if (result == NULL) { kv_destroy(arr); return NULL; }
    for (i = 0; i < size; i++) {
        ptr = PyArray_GETPTR1(result, i);
        obj = kv_pop(arr);
        Py_INCREF(obj);
        *(PyObject **) ptr = obj;
    }
    kv_destroy(arr);
    return (PyObject *) result;
}



static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtree, ptr), READONLY, "Pointer to GEOSSTRtree"},
    {NULL}  /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
    {"query", (PyCFunction) STRtree_query, METH_O,
     "Queries the index for all items whose extents intersect the given search envelope. "
    },
    {NULL}  /* Sentinel */
};

PyTypeObject STRtreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.lib.STRtree",
    .tp_doc = "A query-only R-tree created using the Sort-Tile-Recursive (STR) algorithm.",
    .tp_basicsize = sizeof(STRtree),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = STRtree_new,
    .tp_dealloc = (destructor) STRtree_dealloc,
    .tp_members = STRtree_members,
    .tp_methods = STRtree_methods
};


int init_strtree_type(PyObject *m)
{
    if (PyType_Ready(&STRtreeType) < 0) {
        return -1;
    }

    Py_INCREF(&STRtreeType);
    PyModule_AddObject(m, "STRtree", (PyObject *) &STRtreeType);
    return 0;
}
