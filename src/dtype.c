#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <stdio.h>
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"

#include "dtype.h" 
#include "geos.h"
#include "pygeom.h"


#define NPY_COPY_PYOBJECT_PTR(dst, src) memcpy(dst, src, sizeof(PyObject *))


static NPY_INLINE int
PyGeometry_Check(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&GeometryType);
}

// Functions implementing internal features. Not all of these function
// pointers must be defined for a given type. The required members are
// nonzero, copyswap, copyswapn, setitem, getitem, and cast.
static PyArray_ArrFuncs _PyGeometry_ArrFuncs;


// Those definitions are based on the OBJECT ones of the numpy source
// at numpy/core/src/multiarray/arraytypes.c.src

static void
GEOMETRY_copyswap(PyObject **dst, PyObject **src,
                  int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_OBJECT);
  descr->f->copyswap(dst, src, swap, NULL);
  Py_DECREF(descr);
}

static void
GEOMETRY_copyswapn(PyObject **dst, npy_intp dstride,
                   PyObject **src, npy_intp sstride,
                   npy_intp n, int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_OBJECT);
  descr->f->copyswapn(dst, dstride, src, sstride, n, swap, NULL);
  Py_DECREF(descr);
}

static int
GEOMETRY_setitem(PyObject *op, void *ov, void *NPY_UNUSED(ap))
{
    printf("In GEOMETRY_setitem\n");
    if(!PyGeometry_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                        "Unknown input to GEOMETRY_setitem");
        return -1;
    }
    
    PyObject *obj;

    NPY_COPY_PYOBJECT_PTR(&obj, ov);

    Py_INCREF(op);
    Py_XDECREF(obj);

    NPY_COPY_PYOBJECT_PTR(ov, &op);

    return PyErr_Occurred() ? -1 : 0;
}


static PyObject *
GEOMETRY_getitem(void *ip, void* NPY_UNUSED(arr))
{
  printf("In GEOMETRY_getitem\n");
  PyObject *obj;
  NPY_COPY_PYOBJECT_PTR(&obj, ip);
  if (obj == NULL) {
    Py_RETURN_NONE;
  }
  else {
    Py_INCREF(obj);
    return obj;
  }
}


// int quaternion_elsize = sizeof(GeometryObject);
int geometry_elsize = sizeof(PyObject *);

typedef struct { char c; PyObject * q; } align_test;
int geometry_alignment = offsetof(align_test, q);


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//                                                             //
//  Everything above was preparation for the following set up  //
//                                                             //
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////




//   // Register the quaternion array base type.  Couldn't do this until
//   // after we imported numpy (above)
//   GeometryType.tp_base = &PyGenericArrType_Type;
//   if (PyType_Ready(&GeometryType) < 0) {
//     PyErr_Print();
//     PyErr_SetString(PyExc_SystemError, "Could not initialize GeometryType.");
//     INITERROR;
//   }


void
init_geometry_descriptor(PyObject* np_module)
{
  int npy_registered_geometry;

  PyArray_InitArrFuncs(&_PyGeometry_ArrFuncs);
  _PyGeometry_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)GEOMETRY_copyswap;
  _PyGeometry_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)GEOMETRY_copyswapn;
  _PyGeometry_ArrFuncs.setitem = (PyArray_SetItemFunc*)GEOMETRY_setitem;
  _PyGeometry_ArrFuncs.getitem = (PyArray_GetItemFunc*)GEOMETRY_getitem;

  // The geometry array descr
  geometry_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  geometry_descr->typeobj = &GeometryType;
  geometry_descr->kind = 'O';
  geometry_descr->type = 'g';
  geometry_descr->byteorder = '|';
  //geometry_descr->flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
  geometry_descr->flags = NPY_OBJECT_DTYPE_FLAGS;
  geometry_descr->type_num = 0; // assigned at registration
  geometry_descr->elsize = geometry_elsize;
  geometry_descr->alignment = geometry_alignment;
  geometry_descr->subarray = NULL;
  geometry_descr->fields = NULL;
  geometry_descr->names = NULL;
  geometry_descr->f = &_PyGeometry_ArrFuncs;
  geometry_descr->metadata = NULL;
  geometry_descr->c_metadata = NULL;

  npy_registered_geometry = PyArray_RegisterDataType(geometry_descr);
  if (npy_registered_geometry < 0) {
    return;
  }

}


//     /* Support dtype(qdouble) syntax */
//     if (PyDict_SetItemString(PyQuad_Type.tp_dict, "dtype",
// 			     (PyObject*)&npyquad_descr) < 0) {
//         return;
//     }
