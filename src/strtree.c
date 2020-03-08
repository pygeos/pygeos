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

/* GEOS function that takes a prepared geometry and a regular geometry
 * and returns bool value */

typedef char FuncGEOS_YpY_b(void *context, const GEOSPreparedGeometry *a,
                            const GEOSGeometry *b);




/* get predicate function based on ID.  See strtree.py::BinaryPredicate for
 * lookup table of id to function name */

FuncGEOS_YpY_b *get_predicate_func(int predicate_id) {
    switch (predicate_id) {
        case 1: {  // intersects
            return (FuncGEOS_YpY_b *)GEOSPreparedIntersects_r;
        }
        case 2: { // within
            return (FuncGEOS_YpY_b *)GEOSPreparedWithin_r;
        }
        case 3: { // contains
            return (FuncGEOS_YpY_b *)GEOSPreparedContains_r;
        }
        case 4: { // overlaps
            return (FuncGEOS_YpY_b *)GEOSPreparedOverlaps_r;
        }
        case 5: { // crosses
            return (FuncGEOS_YpY_b *)GEOSPreparedCrosses_r;
        }
        case 6: { // touches
            return (FuncGEOS_YpY_b *)GEOSPreparedTouches_r;
        }
        default: { // unknown predicate
            PyErr_SetString(PyExc_ValueError, "Invalid query predicate");
            return NULL;
        }
    }
}



/* Copy values from arr to a new numpy integer array.
 * The order of values from arr is inverted, because arr is created by pushing
 * values onto the end. */

static PyArrayObject *copy_kvec_to_npy(npy_intp_vec *arr)
{
    npy_intp i;
    npy_intp size = kv_size(*arr);
    npy_intp *ptr;

    npy_intp dims[1] = {size};
    // the following raises a compiler warning based on how the macro is defined
    // in numpy.  There doesn't appear to be anything we can do to avoid it.
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INTP);
    if (result == NULL) {
        return NULL;
    }

    for (i = 0; i<size; i++) {
        ptr = PyArray_GETPTR1(result, i);
        *ptr = kv_A(*arr, i);
    }

    return (PyArrayObject *) result;
}

static void STRtree_dealloc(STRtreeObject *self)
{
    void *context = geos_context[0];
    if (self->ptr != NULL) { GEOSSTRtree_destroy_r(context, self->ptr); }
    Py_XDECREF(self->geometries);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *STRtree_new(PyTypeObject *type, PyObject *args,
                             PyObject *kwds)
{
    int node_capacity;
    PyObject *arr;
    void *tree, *ptr;
    npy_intp n, i;
    long count = 0;
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
            GEOSSTRtree_destroy_r(context, tree);
            return NULL;
        }
        /* skip incase obj was None */
        if (geom == NULL) { continue; }
        /* perform the insert */
        count++;
        GEOSSTRtree_insert_r(context, tree, geom, (void *) i );
    }

    STRtreeObject *self = (STRtreeObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        GEOSSTRtree_destroy_r(context, tree);
        return NULL;
    }
    self->ptr = tree;
    Py_INCREF(arr);
    self->geometries = arr;
    self->count = count;
    return (PyObject *) self;
}


/* Callback to give to strtree_query
 * Given the value returned from each intersecting geometry it inserts that
 * value (the index) into the given size_vector */

void query_callback(void *item, void *user_data)
{
    kv_push(npy_intp, *(npy_intp_vec *)user_data, (npy_intp) item);
}


/* Query the tree based on input geometry and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned. */


/* Evaluate the predicate function against a prepared version of geom
 * for each geometry in the tree specified by indexes in target_indexes.
 * out_indexes is updated in place with the indexes of the geometries in the
 * tree that meet the predicate.
 * Returns the number of geometries that met the predicate.
 * Returns -1 in case of error.
 * */

// TODO: pass in pointer to kvec array and add to it, return count of geoms that are true for predicate
static int evaluate_predicate(FuncGEOS_YpY_b *predicate_func,
                                        GEOSGeometry *geom,
                                        PyArrayObject * tree_geometries,
                                        npy_intp_vec *in_indexes,
                                        npy_intp_vec *out_indexes)
{
    GEOSContextHandle_t context = geos_context[0];
    GEOSGeometry *target_geom;
    const GEOSPreparedGeometry *pgeom;
    npy_intp i, size, index;
    int count = 0;
    npy_intp *geom_ptr;
    PyArrayObject *result;

    // Create prepared geometry
    pgeom = GEOSPrepare_r(context, geom);
    if (pgeom == NULL) {
        return -1;
    }

    size = kv_size(*in_indexes);
    for (i = 0; i < size; i++) {
        // get index for right geometries from in_indexes
        index = kv_A(*in_indexes, i);

        // get GEOS geometry from pygeos geometry at index in tree geometries
        geom_ptr = PyArray_GETPTR1(tree_geometries, index);
        get_geom(*(GeometryObject **) geom_ptr, &target_geom);

        // keep the index value if it passes the predicate
        if (predicate_func(context, pgeom, target_geom)) {
            kv_push(npy_intp, *out_indexes, index);
            count++;
        }
    }

    GEOSPreparedGeom_destroy_r(context, pgeom);

    return count;
}


static PyArrayObject *STRtree_query(STRtreeObject *self, PyObject *args) {
    GEOSContextHandle_t context = geos_context[0];
    GeometryObject *geometry;
    int predicate_id = 0; // default no predicate
    int count = 0;
    GEOSGeometry *geom;
    npy_intp_vec query_indexes, predicate_indexes;
    npy_intp i;
    FuncGEOS_YpY_b *predicate_func;
    PyArrayObject *result;

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }
    if (self->count == 0) {
        npy_intp dims[1] = {0};
        return PyArray_SimpleNew(1, dims, NPY_INTP);
    }

    if (!PyArg_ParseTuple(args, "O!i", &GeometryType, &geometry, &predicate_id)){
        return NULL;
    }

    if (!get_geom(geometry, &geom)) {
        PyErr_SetString(PyExc_TypeError, "Invalid geometry");
        return NULL;
    }

    // query the tree for indices of geometries in the tree with
    // envelopes that intersect the geometry.
    kv_init(query_indexes);
    if (geom != NULL) {
        GEOSSTRtree_query_r(context, self->ptr, geom, query_callback, &query_indexes);
    }

    if (predicate_id == 0 || kv_size(query_indexes) == 0) {
        // No predicate function provided, return all geometry indexes from
        // query.  If array is empty, return an empty numpy array
        result = copy_kvec_to_npy(&query_indexes);
        kv_destroy(query_indexes);
        return (PyArrayObject *) result;
    }

    predicate_func = get_predicate_func(predicate_id);
    if (predicate_func == NULL) {
        // Invalid predicate function
        kv_destroy(query_indexes);
        return NULL;
    }

    kv_init(predicate_indexes);
    count = evaluate_predicate(predicate_func, geom, (PyArrayObject *) self->geometries, &query_indexes, &predicate_indexes);
    if (count == -1) {
        // error performing predicate
        kv_destroy(query_indexes);
        kv_destroy(predicate_indexes);
        return NULL;
    }

    result = copy_kvec_to_npy(&predicate_indexes);

    kv_destroy(query_indexes);
    kv_destroy(predicate_indexes);

    return (PyArrayObject *) result;
}


static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtreeObject, ptr), READONLY, "Pointer to GEOSSTRtree"},
    {"geometries", T_OBJECT_EX, offsetof(STRtreeObject, geometries), READONLY, "Geometries used to construct the GEOSSTRtree"},
    {"count", T_LONG, offsetof(STRtreeObject, count), READONLY, "The number of geometries inside the GEOSSTRtree"},
    {NULL}  /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
    {"query", (PyCFunction) STRtree_query, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search geometry, and optionally tests them "
     "against predicate function if provided. "
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
