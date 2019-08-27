#ifndef _DTYPE_H
#define _DTYPE_H

#include <Python.h>

PyArray_Descr* geometry_descr;

void init_geometry_descriptor(PyObject* np_module);

#endif
