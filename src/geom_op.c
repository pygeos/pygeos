#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>


#include "geos.h"
#include "kvec.h"
#include "pygeom.h"
#include "vector.h"

/* Get a GEOS geometry at position N within a collection of geometries.
 * Only valid for GeometryCollection or Multi* geometries.
 *
 * Negative indexing is supported.
 *
 * Parameters
 * ----------
 * context: GEOS context
 * geom: GEOSGeometry, must be GeometryCollection or Multi*
 * n: position within collection of geometries to return.
 *
 * Returns
 * -------
 * a clone of the geometry, or NULL if geometry does not have geometries
 * or position n resolves to an index that is out of bounds.
 */

void *GetGeometryN(void *context, void *geom, int n)
{
    int size, i;
    size = GEOSGetNumGeometries_r(context, geom);
    if (size == -1) {
        return NULL;
    }
    if (n < 0) {
        /* Negative indexing: we get it for free */
        i = size + n;
    } else {
        i = n;
    }
    if ((i < 0) | (i >= size)) {
        /* Important, could give segfaults else */
        return NULL;
    }
    void *ret = (void *) GEOSGetGeometryN_r(context, geom, i);
    /* Create a copy of the obtained geometry */
    if (ret != NULL) {
        ret = GEOSGeom_clone_r(context, ret);
    }
    return ret;
}

/* Add all geometries in a GEOS geometry to a resizable vector.
 * Only valid for GeometryCollection or Multi* geometries.
 *
 * Parameters
 * ----------
 * context: GEOS context
 * geom: GEOSGeometry, must be GeometryCollection or Multi*
 * geometries: resizable vector of GEOSGeometry (for parts)
 */
static int GetGeometries(void *context, void *geom, geom_obj_vec *geometries)
{
    GEOSGeometry *ret;
    int count, i;

    if (geom == NULL) {
        return NULL;
    }

    count = GEOSGetNumGeometries_r(context, geom);

    if (count == -1) {
        return -1;
    }
    for (i=0; i<count; i++) {
        // This also clones the underlying geometry part
        kv_push(GEOSGeometry *, *geometries, GetGeometryN(context, geom, i));
    }

    return count;
}

/* Python interface function to get all parts for each geometry in an
 * array of geometries.  If a given geometry is NULL, empty, or has no parts,
 * it will be returned as is (as a clone, if a geometry).
 *
 * Parameters
 * ----------
 * arr: 1d array of pygeos Geometry objects
 *
 * Returns
 * -------
 * tuple of (indexes, geometries)
 *
 */
PyObject *PyGetParts(PyObject *self, PyObject *arr) {
    GEOSContextHandle_t context = geos_context[0];
    PyArrayObject *pg_geoms;
    GeometryObject *pg_geom;
    GEOSGeometry *geom, *part;
    npy_intp_vec src_indexes;
    geom_obj_vec geom_parts;
    npy_intp i, j, n, size;
    int geom_type;
    PyArrayObject *out_indexes, *out_pgeoms;
    PyObject *result;

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }

    pg_geoms = (PyArrayObject *) arr;
    if (!PyArray_ISOBJECT(pg_geoms)) {
        PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
        return NULL;
    }

    if (PyArray_NDIM(pg_geoms) != 1) {
        PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
        return NULL;
    }

    kv_init(src_indexes);
    kv_init(geom_parts);
    n = PyArray_SIZE(pg_geoms);

    for(i = 0; i < n; i++) {
        // get pygeos geometry from input geometry array
        pg_geom = *(GeometryObject **) PyArray_GETPTR1(pg_geoms, i);

        // get GEOSGeometry from pygeos geometry
        if (!get_geom(pg_geom, &geom)) {
            PyErr_SetString(PyExc_TypeError, "Invalid geometry");
            return NULL;
        }
        if (geom == NULL) {
            continue;
        }
        if (GEOSisEmpty_r(context, geom)) {
            continue;
        }

        geom_type = GEOSGeomTypeId_r(context, geom);
        if (geom_type == -1) {
            RAISE_ILLEGAL_GEOS;
        }
        if (geom_type == 7) {
            // geometry collection
            // TODO: recursively expand geometry
            PyErr_SetString(PyExc_TypeError, "GeometryCollection: not yet implemented");
        }
        else if (geom_type >= 4 && geom_type <= 6) {
            // Multi* geometry
            size = GetGeometries(context, geom, &geom_parts);
            if (size == -1) {
                PyErr_SetString(PyExc_RuntimeError, "Error extracting parts from geometry");
                return NULL;
            }
            for (j=0; j<size; j++) {
                kv_push(npy_intp, src_indexes, i);
            }
        }
        else {
            // singular geometry
            part = GEOSGeom_clone_r(context, geom);
            kv_push(GEOSGeometry *, geom_parts, part);
            kv_push(npy_intp, src_indexes, i);
        }
    }

    // make tuple by converting vectors to ndarrays
    result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, (PyObject *)npy_intp_vec_to_npy_arr(&src_indexes));
    PyTuple_SetItem(result, 1, (PyObject *)geom_obj_vec_to_npy_arr(&geom_parts));

    kv_destroy(src_indexes);
    kv_destroy(geom_parts);

    return result;
}
