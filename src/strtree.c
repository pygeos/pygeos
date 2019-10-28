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


void strtree_dealloc_callback(void *item, void *vec)
{
    STRtreeElem *elem = item;
    Py_XDECREF(elem->geometry);
    free(elem);
}

void GEOSSTRtree_dealloc(GEOSContextHandle_t context, GEOSSTRtree *tree)
{
    if (tree != NULL) {
        /* Decrease the refcount of each GeometryObject in the STRTree */
        GEOSSTRtree_iterate_r(context, tree, strtree_dealloc_callback, NULL);
        GEOSSTRtree_destroy_r(context, tree);
    }
}

static void STRtree_dealloc(STRtreeObject *self)
{
    void *context = geos_context[0];
    GEOSSTRtree_dealloc(context, self->ptr);
    Py_XDECREF(self->geometries);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

void strtree_insert(GEOSContextHandle_t context, GEOSSTRtree *tree, GEOSGeometry *geometry, GeometryObject *obj, npy_intp i)
{
    STRtreeElem *elem;
    elem = malloc(sizeof(STRtreeElem));
    elem->i = i;
    elem->geometry = (PyObject *) obj;
    Py_INCREF(obj);   /* STRtree holds a reference to each GeometryObject */
    GEOSSTRtree_insert_r(context, tree, geometry, elem );
}

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
        /* fail and cleanup incase obj was no geometry */
        if (!get_geom(obj, &geom)) {
            GEOSSTRtree_dealloc(context, tree);
            return NULL;
        }
        /* skip incase obj was None */
        if (geom == NULL) { continue; }
        /* perform the insert */
        strtree_insert(context, tree, geom, obj, i);
    }

    STRtreeObject *self = (STRtreeObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        GEOSSTRtree_destroy_r(context, tree);
        return NULL;
    }
    self->ptr = tree;
    Py_INCREF(arr);
    self->geometries = arr;
    return (PyObject *) self;
}

/* Callback to give to strtree_query
 * Given the value returned from each intersecting geometry it inserts that
 * value (typically an index) into the given size_vector */

void strtree_query_callback(void *item, void *user_data)
{
    STRtreeElem *elem = item;
    kv_push(npy_intp, *(npy_intp_vec *)user_data, elem->i);
}

static PyObject *STRtree_query(STRtreeObject *self, PyObject *envelope) {
    GEOSContextHandle_t context = geos_context[0];
    GEOSGeometry *geom;
    npy_intp_vec arr; // Resizable array for matches for each geometry
    npy_intp i, size;
    npy_intp *ptr;

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }
    if (!get_geom((GeometryObject *) envelope, &geom)) {
        PyErr_SetString(PyExc_TypeError, "Invalid geometry");
        return NULL;
    }

    kv_init(arr);
    if (geom != NULL) {
        GEOSSTRtree_query_r(context, self->ptr, geom, strtree_query_callback, &arr);
    }

    /* create an index array with the appropriate dimensions */
    size = kv_size(arr);
    npy_intp dims[1] = {size};
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INTP);
    if (result == NULL) { kv_destroy(arr); return NULL; }
    /* insert values starting from array end to preserve order */
    for (i = size - 1; i >= 0; i--) {
        ptr = PyArray_GETPTR1(result, i);
        *ptr = kv_pop(arr);
    }
    kv_destroy(arr);
    return (PyObject *) result;
}



static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtreeObject, ptr), READONLY, "Pointer to GEOSSTRtree"},
    {"geometries", T_OBJECT_EX, offsetof(STRtreeObject, geometries), READONLY, "Geometries used to construct the GEOSSTRtree"},
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
    .tp_basicsize = sizeof(STRtreeObject),
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
