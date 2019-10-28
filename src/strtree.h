#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>
#include "geos.h"


typedef struct {
    PyObject_HEAD
    void *ptr;
    PyObject *geometries;
} STRtree;

/* A resizable vector with numpy indices */
typedef struct
{
    size_t n, m;
    npy_intp *a;
} npy_intp_vec;

/* An element in the tree */
typedef struct {
    npy_intp i;
    PyObject *geometry;
} STRtreeElem;

PyTypeObject STRtreeType;

int init_strtree_type(PyObject *m);

#endif
