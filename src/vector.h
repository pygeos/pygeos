#ifndef _VECTOR_H
#define _VECTOR_H

#include <Python.h>
#include <numpy/ndarraytypes.h>

#include "geos.h"
#include "kvec.h"
#include "pygeom.h"

/* A resizable vector with numpy indices.
 * Wraps the vector implementation in kvec.h as a type.
 */
typedef struct {
  size_t n, m;
  npy_intp* a;
} npy_intp_vec;

/* A resizable vector with pointers to pygeos GeometryObjects.
 * Wraps the vector implementation in kvec.h as a type.
 */
typedef struct {
  size_t n, m;
  GeometryObject** a;
} geom_obj_vec;

/* Copy values from arr to a new numpy integer array.
 *
 * Parameters
 * ----------
 * arr: dynamic vector array to convert to ndarray
 */
extern PyArrayObject* npy_intp_vec_to_npy_arr(npy_intp_vec* arr);

#endif
