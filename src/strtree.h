#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>
#include "geos.h"


typedef struct {
    PyObject_HEAD
    void *ptr;
    PyObject *geometries;
} STRtree;


PyTypeObject STRtreeType;

int init_strtree_type(PyObject *m);

#endif
