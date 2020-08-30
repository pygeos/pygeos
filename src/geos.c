#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "geos.h"

/* This initializes a globally accessible GEOSException object */
PyObject *geos_exception[1] = {NULL};

int init_geos(PyObject *m)
{
    geos_exception[0] = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
    PyModule_AddObject(m, "GEOSException", geos_exception[0]);
    return 0;
}

/* Returns 1 if geometry is an empty point, 0 otherwise, 2 on error.
*/
char is_point_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom) {
    int geom_type;

    geom_type = GEOSGeomTypeId_r(ctx, geom);
    if (geom_type == GEOS_POINT) {
        return GEOSisEmpty_r(ctx, geom);
    } else if (geom_type == -1) {
        return 2;  // GEOS exception
    } else {
        return 0;  // No empty point
    }
}

/* Returns 1 if a multipoint has an empty point, 0 otherwise, 2 on error.
*/
char multipoint_has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom) {    
    int n, i;
    char is_empty;
    const GEOSGeometry *sub_geom;
    
    n = GEOSGetNumGeometries_r(ctx, geom);
    if (n == -1) { return 2; }
    for(i = 0; i < n; i++) {
        sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
        if (sub_geom == NULL) { return 2; }
        is_empty = GEOSisEmpty_r(ctx, sub_geom);
        if (is_empty != 0) {
            // If empty encountered, or on exception, return:
            return is_empty;
        }
    }
    return 0;
}

/* Returns 1 if a geometrycollection has an empty point, 0 otherwise, 2 on error.
Checks recursively (geometrycollections may contain multipoints / geometrycollections)
*/
char geometrycollection_has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom) {    
    int n, i;
    char has_empty;
    const GEOSGeometry *sub_geom;
    
    n = GEOSGetNumGeometries_r(ctx, geom);
    if (n == -1) { return 2; }
    for(i = 0; i < n; i++) {
        sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
        if (sub_geom == NULL) { return 2; }
        has_empty = has_point_empty(ctx, (GEOSGeometry *) sub_geom);
        if (has_empty != 0) {
            // If empty encountered, or on exception, return:
            return has_empty;
        }
    }
    return 0;
}

/* Returns 1 if geometry is / has an empty point, 0 otherwise, 2 on error.
*/
char has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom) {
    int geom_type;

    geom_type = GEOSGeomTypeId_r(ctx, geom);
    if (geom_type == GEOS_POINT) {
        return GEOSisEmpty_r(ctx, geom);
    } else if (geom_type == GEOS_MULTIPOINT) {
        return multipoint_has_point_empty(ctx, geom);
    } else if (geom_type == GEOS_GEOMETRYCOLLECTION) {
        return geometrycollection_has_point_empty(ctx, geom);
    } else if (geom_type == -1) {
        return 2;  // GEOS exception
    } else {
        return 0;  // No empty point
    }
}

/* Creates a POINT (nan, nan[, nan)] from a POINT EMPTY template

   Returns NULL on error
*/
GEOSGeometry *point_empty_to_nan(GEOSContextHandle_t ctx, GEOSGeometry *geom) {
    int j, ndim;
    GEOSCoordSequence *coord_seq;
    GEOSGeometry *result;

    ndim = GEOSGeom_getCoordinateDimension_r(ctx, geom);
    if (ndim == 0) { return NULL; }
    
    coord_seq = GEOSCoordSeq_create_r(ctx, 1, ndim);
    if (coord_seq == NULL) { return NULL; }
    for (j = 0; j < ndim; j++) {
        if (!GEOSCoordSeq_setOrdinate_r(ctx, coord_seq, 0, j, Py_NAN)) {
            GEOSCoordSeq_destroy_r(ctx, coord_seq);
            return NULL;
        }
    }
    result = GEOSGeom_createPoint_r(ctx, coord_seq);
    if (result == NULL) {
        GEOSCoordSeq_destroy_r(ctx, coord_seq); 
        return NULL;
    }
    GEOSSetSRID_r(ctx, result, GEOSGetSRID_r(ctx, geom));
    return result;
}

/* Returns 1 if geom is a point with only nan coordinates, 0 otherwise, 2 on error.
*/
char is_point_nan(GEOSContextHandle_t ctx, GEOSGeometry *geom) {
    int geom_type;
    char is_empty;
    int j, ndim;
    double coord;
    const GEOSCoordSequence *coord_seq;

    geom_type = GEOSGeomTypeId_r(ctx, geom);
    if (geom_type >= 1) {
        return 0;
    } else if (geom_type == -1) {
        return 2;  // GEOS exception
    }

    is_empty = GEOSisEmpty_r(ctx, geom);
    if (is_empty == 1) {
        return 0;
    } else if (is_empty == 2) {
        return 2;  // GEOS exception
    }

    ndim = GEOSGeom_getCoordinateDimension_r(ctx, geom);
    if (ndim == 0) { return 2; }

    coord_seq = GEOSGeom_getCoordSeq_r(ctx, geom);
    for (j = 0; j < ndim; j++) {
        if (!GEOSCoordSeq_getOrdinate_r(ctx, coord_seq, 0, j, &coord)) {
            return 2;
        }
        if (!isnan(coord)) {
            // Coordinate is not NaN; do not replace the geometry
            return 0;
        }
    }
    return 1;
}

/* Creates a POINT Z EMPTY from a POINT (nan, nan[, nan]) template

   Returns NULL on error
*/
GEOSGeometry *point_nan_to_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom) {
    GEOSGeometry *result;

    result = GEOSGeom_createEmptyPoint_r(ctx);
    if (result == NULL) {
        return NULL;
    }
    GEOSSetSRID_r(ctx, result, GEOSGetSRID_r(ctx, geom));
    return result;
}


/* Checks whether the geometry is a multipoint with an empty point in it
 *
 * According to https://github.com/libgeos/geos/issues/305, this check is not
 * necessary for GEOS 3.7.3, 3.8.2, or 3.9. When these versions are out, we 
 * should add version conditionals and test.
 * 
 * The return value is one of:
 * - PGERR_SUCCESS 
 * - PGERR_WKT_INCOMPATIBLE
 * - PGERR_GEOS_EXCEPTION
 */
char check_to_wkt_compatible(GEOSContextHandle_t ctx, GEOSGeometry *geom) {    
    char geom_type, is_empty;

    geom_type = GEOSGeomTypeId_r(ctx, geom);
    if (geom_type == -1) { return PGERR_GEOS_EXCEPTION; }
    if (geom_type != GEOS_MULTIPOINT) { return PGERR_SUCCESS; }

    is_empty = multipoint_has_point_empty(ctx, geom);
    if (is_empty == 0) {
        return PGERR_SUCCESS;
    } else if (is_empty == 1) {
        return PGERR_WKT_INCOMPATIBLE;
    } else {
        return PGERR_GEOS_EXCEPTION;
    }
}

/* Define GEOS error handlers. See GEOS_INIT / GEOS_FINISH macros in geos.h*/
void geos_error_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}

void geos_notice_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}
