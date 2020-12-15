#ifndef _PYGEOSCOORDS_H
#define _PYGEOSCOORDS_H

#include <Python.h>

#include "geos.h"

int get_bounds(GEOSContextHandle_t ctx, GEOSGeometry* geom, double* xmin, double* ymin,
               double* xmax, double* ymax);
GEOSGeometry* create_box(GEOSContextHandle_t ctx, double xmin, double ymin, double xmax,
                         double ymax);

extern PyObject* PyCountCoords(PyObject* self, PyObject* args);
extern PyObject* PyGetCoords(PyObject* self, PyObject* args);
extern PyObject* PySetCoords(PyObject* self, PyObject* args);

#endif
