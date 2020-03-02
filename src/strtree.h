#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>
#include "geos.h"


/* A resizable vector with numpy indices */
typedef struct
{
    size_t n, m;
    npy_intp *a;
} npy_intp_vec;


/* A resizable vector with GEOSGeometry pointers */
typedef struct
{
    size_t n, m;
    GEOSGeometry **a;
} goes_geom_vec;

typedef struct {
    PyObject_HEAD
    void *ptr;
    goes_geom_vec _geoms;
} STRtreeObject;


extern PyTypeObject STRtreeType;

extern int init_strtree_type(PyObject *m);

#endif
