#ifndef _PYGEOM_H
#define _PYGEOM_H

#include <Python.h>
#include "geos.h"

typedef struct
{
    PyObject_HEAD void *ptr;
} GeometryObject;

extern PyTypeObject GeometryType;

/* Initializes a new geometry object from a GEOSGeometry */
extern PyObject *GeometryObject_FromGEOS(PyTypeObject *type, GEOSGeometry *ptr);

/* Get a GEOSGeometry from a GeometryObject */
extern char get_geom(GeometryObject *obj, GEOSGeometry **out);

extern int init_geom_type(PyObject *m);


/*****  Prepared Geometries *****/

typedef struct
{
    PyObject_HEAD void *ptr;
} PreparedGeometryObject;

extern PyTypeObject PreparedGeometryType;

/* Initializes a new geometry object from a GEOSPreparedGeometry */
extern PyObject *PreparedGeometryObject_FromGEOSPreparedGeometry(PyTypeObject *type, GEOSPreparedGeometry *ptr);

/* Get a GEOSPreparedGeometry from a PreparedGeometryObject */
extern char get_geom_prepared(PreparedGeometryObject *obj, GEOSPreparedGeometry **out);



#endif
