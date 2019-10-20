#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>
#include "geos.h"


typedef struct {
    PyObject_HEAD
    void *ptr;
} STRtree;

/* A resizable vector with geometry objects */
typedef struct
{
    size_t n, m;
    PyObject **a;
} geom_array;


PyTypeObject STRtreeType;

int init_strtree_type(PyObject *m);

#endif
