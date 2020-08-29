#ifndef _PYGEOM_H
#define _PYGEOM_H

#include <Python.h>
#include "geos.h"


typedef struct {
    PyObject_HEAD
    void *ptr;
} GeometryObject;


extern PyTypeObject GeometryType;

/* Initializes a new geometry object */
extern PyObject *GeometryObject_FromGEOS(PyTypeObject *type, GEOSGeometry *ptr);
/* Get a GEOSGeometry from a GeometryObject */
extern char get_geom(GeometryObject *obj, GEOSGeometry **out);

extern char is_point_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom);
extern GEOSGeometry *point_empty_to_nan(GEOSContextHandle_t ctx, GEOSGeometry *geom);
extern char is_point_nan(GEOSContextHandle_t ctx, GEOSGeometry *geom);
extern GEOSGeometry *point_nan_to_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom);

extern int init_geom_type(PyObject *m);

#endif
