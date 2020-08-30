#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "pygeom.h"
#include "geos.h"

char* repr_fmt = "<pygeos.Geometry %s>";

/* Initializes a new geometry object */
PyObject *GeometryObject_FromGEOS(PyTypeObject *type, GEOSGeometry *ptr)
{
    if (ptr == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    GeometryObject *self = (GeometryObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    } else {
        self->ptr = ptr;
        return (PyObject *) self;
    }
}

static void GeometryObject_dealloc(GeometryObject *self)
{
    if (self->ptr != NULL) {
        GEOS_INIT;
        GEOSGeom_destroy_r(ctx, self->ptr);
        GEOS_FINISH;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef GeometryObject_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(GeometryObject, ptr), READONLY, "pointer to GEOSGeometry"},
    {NULL}  /* Sentinel */
};

static PyObject *GeometryObject_ToWKT(GeometryObject *obj, char *format)
{
    char *wkt;
    PyObject *result;
    if (obj->ptr == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    GEOS_INIT;

    errstate = check_to_wkt_compatible(ctx, obj->ptr);
    if (errstate != PGERR_SUCCESS) { goto finish; }

    GEOSWKTWriter *writer = GEOSWKTWriter_create_r(ctx);
    if (writer == NULL) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }

    char trim = 1;
    int precision = 3;
    int dimension = 3;
    int use_old_3d = 0;
    GEOSWKTWriter_setRoundingPrecision_r(ctx, writer, precision);
    GEOSWKTWriter_setTrim_r(ctx, writer, trim);
    GEOSWKTWriter_setOutputDimension_r(ctx, writer, dimension);
    GEOSWKTWriter_setOld3D_r(ctx, writer, use_old_3d);

    // Check if the above functions caused a GEOS exception
    if (last_error[0] != 0) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }

    wkt = GEOSWKTWriter_write_r(ctx, writer, obj->ptr);
    result = PyUnicode_FromFormat(format, wkt);
    GEOSFree_r(ctx, wkt);
    GEOSWKTWriter_destroy_r(ctx, writer);

    finish:
        GEOS_FINISH;
        if (errstate == PGERR_SUCCESS) {
            return result;
        } else {
            return NULL;
        }
}

static PyObject *GeometryObject_repr(GeometryObject *self)
{
    PyObject *result = GeometryObject_ToWKT(self, repr_fmt);
    // we never want a repr() to fail; that can be very confusing
    if (result == NULL) {
        PyErr_Clear();
        return PyUnicode_FromFormat(repr_fmt, "Exception in WKT writer");
    }
    return result;
}

static PyObject *GeometryObject_str(GeometryObject *self)
{
    return GeometryObject_ToWKT(self, "%s");
}

static Py_hash_t GeometryObject_hash(GeometryObject *self)
{
    unsigned char *wkb;
    size_t size;
    Py_hash_t x;

    if (self->ptr == NULL) {
        return -1;
    }

    GEOS_INIT;
    GEOSWKBWriter *writer = GEOSWKBWriter_create_r(ctx);
    if (writer == NULL) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }

    GEOSWKBWriter_setOutputDimension_r(ctx, writer, 3);
    GEOSWKBWriter_setIncludeSRID_r(ctx, writer, 1);
    wkb = GEOSWKBWriter_write_r(ctx, writer, self->ptr, &size);
    GEOSWKBWriter_destroy_r(ctx, writer);
    if (wkb == NULL) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }
    x = PyHash_GetFuncDef()->hash(wkb, size);
    if (x == -1) {
        x = -2;
    } else {
        x ^= 374761393UL;  // to make the result distinct from the actual WKB hash //
    }
    GEOSFree_r(ctx, wkb);

    finish:
        GEOS_FINISH;
        if (errstate == PGERR_SUCCESS) {
            return x;
        } else {
            return -1;
        }
}

static PyObject *GeometryObject_richcompare(GeometryObject *self, PyObject *other, int op) {
  PyObject *result = NULL;
  GEOS_INIT;
  if (Py_TYPE(self)->tp_richcompare != Py_TYPE(other)->tp_richcompare) {
      result = Py_NotImplemented;
  } else {
      GeometryObject *other_geom = (GeometryObject *) other;
      switch (op) {
      case Py_LT:
        result = Py_NotImplemented;
        break;
      case Py_LE:
        result = Py_NotImplemented;
        break;
      case Py_EQ:
        result = GEOSEqualsExact_r(ctx, self->ptr, other_geom->ptr, 0) ? Py_True : Py_False;
        break;
      case Py_NE:
        result = GEOSEqualsExact_r(ctx, self->ptr, other_geom->ptr, 0) ? Py_False : Py_True;
        break;
      case Py_GT:
        result = Py_NotImplemented;
        break;
      case Py_GE:
        result = Py_NotImplemented;
        break;
    }
  }
  GEOS_FINISH;
  Py_XINCREF(result);
  return result;
}

static PyObject *GeometryObject_FromWKT(PyTypeObject *type, PyObject *value)
{
    PyObject *result = NULL;
    const char *wkt;
    GEOSGeometry *geom;
    GEOSWKTReader *reader;

    /* Cast the PyObject str to char* */
    if (PyUnicode_Check(value)) {
        wkt = PyUnicode_AsUTF8(value);
        if (wkt == NULL) { return NULL; }
    } else {
        PyErr_Format(PyExc_TypeError, "Expected bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }

    GEOS_INIT;

    reader = GEOSWKTReader_create_r(ctx);
    if (reader == NULL) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }
    geom = GEOSWKTReader_read_r(ctx, reader, wkt);
    GEOSWKTReader_destroy_r(ctx, reader);
    if (geom == NULL) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }
    result = GeometryObject_FromGEOS(type, geom);
    if (result == NULL) {
        GEOSGeom_destroy_r(ctx, geom);
        PyErr_Format(PyExc_RuntimeError, "Could not instantiate a new Geometry object");
    }

    finish:
        GEOS_FINISH;
        if (errstate == PGERR_SUCCESS) {
            return result;
        } else {
            return NULL;
        }
}

static PyObject *GeometryObject_new(PyTypeObject *type, PyObject *args,
                                    PyObject *kwds)
{
    PyObject *value;

    if (!PyArg_ParseTuple(args, "O", &value)) {
        return NULL;
    }
    else if (PyUnicode_Check(value)) {
        return GeometryObject_FromWKT(type, value);
    }
    else {
        PyErr_Format(PyExc_TypeError, "Expected string, got %s", value->ob_type->tp_name);
        return NULL;
    }
}

static PyMethodDef GeometryObject_methods[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject GeometryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.lib.GEOSGeometry",
    .tp_doc = "Geometry type",
    .tp_basicsize = sizeof(GeometryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeometryObject_new,
    .tp_dealloc = (destructor) GeometryObject_dealloc,
    .tp_members = GeometryObject_members,
    .tp_methods = GeometryObject_methods,
    .tp_repr = (reprfunc) GeometryObject_repr,
    .tp_hash = (hashfunc) GeometryObject_hash,
    .tp_richcompare = (richcmpfunc) GeometryObject_richcompare,
    .tp_str = (reprfunc) GeometryObject_str,
};


/* Get a GEOSGeometry pointer from a GeometryObject, or NULL if the input is
Py_None. Returns 0 on error, 1 on success. */
char get_geom(GeometryObject *obj, GEOSGeometry **out) {
    PyTypeObject *type = ((PyObject *)obj)->ob_type;
    if ((type != &GeometryType) & (type->tp_base != &GeometryType)) {
        if ((PyObject *) obj == Py_None) {
            *out = NULL;
            return 1;
        } else {
            return 0;
        }
    } else {
        *out = obj->ptr;
        return 1;
    }
}


/* Returns 1 if geom is an empty point, 0 otherwise, 2 on error.
*/
char is_point_empty(GEOSContextHandle_t ctx, GEOSGeometry *geom) {
    int geom_type;

    geom_type = GEOSGeomTypeId_r(ctx, geom);
    if (geom_type >= 1) {
        return 0;
    } else if (geom_type == -1) {
        return 2;  // GEOS exception
    }

    return GEOSisEmpty_r(ctx, geom);
}

/* Transforms a POINT EMPTY into POINT (nan, nan[, nan]) for serialization

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

/* Transforms a POINT (nan, nan[, nan]) into POINT Z EMPTY for deserialization

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

int
init_geom_type(PyObject *m)
{
    if (PyType_Ready(&GeometryType) < 0) {
        return -1;
    }

    Py_INCREF(&GeometryType);
    PyModule_AddObject(m, "Geometry", (PyObject *) &GeometryType);
    return 0;
}
