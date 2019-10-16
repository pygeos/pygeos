#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "strtree.h"
#include "geos.h"


static PyObject *STRtree_new(PyTypeObject *type, PyObject *args,
                             PyObject *kwds)
{
    int node_capacity;
    void *tree;
    GEOSContextHandle_t context = geos_context[0];

    if (!PyArg_ParseTuple(args, "i", &node_capacity)) {
        return NULL;
    }
    tree = GEOSSTRtree_create_r(context, (size_t) node_capacity);
    if (tree == NULL) {
        return NULL;
    }
    STRtree *self = (STRtree *) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    } else {
        self->ptr = tree;
        return (PyObject *) self;
    }
}

static void STRtree_dealloc(STRtree *self)
{
    void *context = geos_context[0];
    if (self->ptr != NULL) {
        GEOSSTRtree_destroy_r(context, self->ptr);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtree, ptr), READONLY, "pointer to GEOSSTRtree"},
    {NULL}  /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
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
