#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>

#include "geos.h"
#include "pygeom.h"

/* A resizable vector with numpy indices */
typedef struct {
  size_t n, m;
  npy_intp* a;
} npy_intp_vec;

/* A resizable vector with addresses of geometries within tree geometries array */
typedef struct {
  size_t n, m;
  GeometryObject*** a;
} tree_geom_vec_t;

typedef struct {
  PyObject_HEAD void* ptr;
  npy_intp count;           // count of geometries added to the tree
  size_t _size;             // size of _geoms array (same as original size of input array)
  GeometryObject** _geoms;  // array of input geometries
} STRtreeObject;

extern PyTypeObject STRtreeType;

extern int init_strtree_type(PyObject* m);

#endif
