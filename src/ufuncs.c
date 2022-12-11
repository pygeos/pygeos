#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <math.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL pygeos_UFUNC_API
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/ufuncobject.h>

#include "fast_loop_macros.h"
#include "geos.h"
#include "pygeom.h"

#define OUTPUT_Y                                         \
  PyObject* ret = GeometryObject_FromGEOS(ret_ptr, ctx); \
  PyObject** out = (PyObject**)op1;                      \
  Py_XDECREF(*out);                                      \
  *out = ret

#define OUTPUT_Y_I(I, RET_PTR)                              \
  PyObject* ret##I = GeometryObject_FromGEOS(RET_PTR, ctx); \
  PyObject** out##I = (PyObject**)op##I;                    \
  Py_XDECREF(*out##I);                                      \
  *out##I = ret##I

// Fail if inputs output multiple times on the same place in memory. That would
// lead to segfaults as the same GEOSGeometry would be 'owned' by multiple PyObjects.
#define CHECK_NO_INPLACE_OUTPUT(N)                                             \
  if ((steps[N] == 0) && (dimensions[0] > 1)) {                                \
    PyErr_Format(PyExc_NotImplementedError,                                    \
                 "Zero-strided output detected. Ufunc mode with args[0]=%p, "  \
                 "args[N]=%p, steps[0]=%ld, steps[N]=%ld, dimensions[0]=%ld.", \
                 args[0], args[N], steps[0], steps[N], dimensions[0]);         \
    return;                                                                    \
  }

#define CHECK_ALLOC(ARR)                                             \
  if (ARR == NULL) {                                                 \
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory"); \
    return;                                                          \
  }

static void geom_arr_to_npy(GEOSGeometry** array, char* ptr, npy_intp stride,
                            npy_intp count) {
  npy_intp i;
  PyObject* ret;
  PyObject** out;

  GEOS_INIT;

  for (i = 0; i < count; i++, ptr += stride) {
    ret = GeometryObject_FromGEOS(array[i], ctx);
    out = (PyObject**)ptr;
    Py_XDECREF(*out);
    *out = ret;
  }

  GEOS_FINISH;
}

/* Define the geom -> bool functions (Y_b) */
static void* is_empty_data[1] = {GEOSisEmpty_r};
/* the GEOSisSimple_r function fails on geometrycollections */
static char GEOSisSimpleAllTypes_r(void* context, void* geom) {
  int type = GEOSGeomTypeId_r(context, geom);
  if (type == -1) {
    return 2;  // Predicates use a return value of 2 for errors
  } else if (type == 7) {
    return 0;
  } else {
    return GEOSisSimple_r(context, geom);
  }
}
static void* is_simple_data[1] = {GEOSisSimpleAllTypes_r};
static void* is_ring_data[1] = {GEOSisRing_r};
static void* has_z_data[1] = {GEOSHasZ_r};
/* the GEOSisClosed_r function fails on non-linestrings */
static char GEOSisClosedAllTypes_r(void* context, void* geom) {
  int type = GEOSGeomTypeId_r(context, geom);
  if (type == -1) {
    return 2;  // Predicates use a return value of 2 for errors
  } else if ((type == 1) || (type == 2) || (type == 5)) {
    return GEOSisClosed_r(context, geom);
  } else {
    return 0;
  }
}
static void* is_closed_data[1] = {GEOSisClosedAllTypes_r};
static void* is_valid_data[1] = {GEOSisValid_r};

#if GEOS_SINCE_3_7_0
static char GEOSGeom_isCCW_r(void* context, void* geom) {
  const GEOSCoordSequence* coord_seq;
  char is_ccw = 2;  // return value of 2 means GEOSException
  int i;

  // Return False for non-linear geometries
  i = GEOSGeomTypeId_r(context, geom);
  if (i == -1) {
    return 2;
  }
  if ((i != GEOS_LINEARRING) && (i != GEOS_LINESTRING)) {
    return 0;
  }

  // Return False for lines with fewer than 4 points
  i = GEOSGeomGetNumPoints_r(context, geom);
  if (i == -1) {
    return 2;
  }
  if (i < 4) {
    return 0;
  }

  // Get the coordinatesequence and call isCCW()
  coord_seq = GEOSGeom_getCoordSeq_r(context, geom);
  if (coord_seq == NULL) {
    return 2;
  }
  if (!GEOSCoordSeq_isCCW_r(context, coord_seq, &is_ccw)) {
    return 2;
  }
  return is_ccw;
}
static void* is_ccw_data[1] = {GEOSGeom_isCCW_r};
#endif

typedef char FuncGEOS_Y_b(void* context, void* a);
static char Y_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};
static void Y_b_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_Y_b* func = (FuncGEOS_Y_b*)data;
  GEOSGeometry* in1 = NULL;
  char ret;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    /* get the geometry; return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (in1 == NULL) {
      /* in case of a missing value: return 0 (False) */
      ret = 0;
    } else {
      /* call the GEOS function */
      ret = func(ctx, in1);
      /* finish for illegal values */
      if (ret == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction Y_b_funcs[1] = {&Y_b_func};

/* Define the object -> bool functions (O_b) which do not raise on non-geom objects*/
static char IsMissing(void* context, PyObject* obj) {
  GEOSGeometry* g = NULL;
  if (!get_geom((GeometryObject*)obj, &g)) {
    return 0;
  };
  return g == NULL;  // get_geom sets g to NULL for None input
}
static void* is_missing_data[1] = {IsMissing};
static char IsGeometry(void* context, PyObject* obj) {
  GEOSGeometry* g = NULL;
  if (!get_geom((GeometryObject*)obj, &g)) {
    return 0;
  }
  return g != NULL;
}
static void* is_geometry_data[1] = {IsGeometry};
static char IsValidInput(void* context, PyObject* obj) {
  GEOSGeometry* g = NULL;
  return get_geom((GeometryObject*)obj, &g);
}
static void* is_valid_input_data[1] = {IsValidInput};
typedef char FuncGEOS_O_b(void* context, PyObject* obj);
static char O_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};
static void O_b_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_O_b* func = (FuncGEOS_O_b*)data;
  GEOS_INIT_THREADS;
  UNARY_LOOP { *(npy_bool*)op1 = func(ctx, *(PyObject**)ip1); }
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction O_b_funcs[1] = {&O_b_func};

/* Define the geom, geom -> bool functions (YY_b) */
static void* equals_data[1] = {GEOSEquals_r};
typedef char FuncGEOS_YY_b(void* context, void* a, void* b);
static char YY_b_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_BOOL};
static void YY_b_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_YY_b* func = (FuncGEOS_YY_b*)data;
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  char ret;

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if ((in1 == NULL) || (in2 == NULL)) {
      /* in case of a missing value: return 0 (False) */
      ret = 0;
    } else {
      /* call the GEOS function */
      ret = func(ctx, in1, in2);
      /* return for illegal values */
      if (ret == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction YY_b_funcs[1] = {&YY_b_func};

/* Define the geom, geom -> bool functions (YY_b) prepared */
static void* contains_func_tuple[2] = {GEOSContains_r, GEOSPreparedContains_r};
static void* contains_data[1] = {contains_func_tuple};
static char GEOSContainsProperly(void* context, void* g1, void* g2) {
  const GEOSPreparedGeometry* prepared_geom_tmp = NULL;
  char ret;

  prepared_geom_tmp = GEOSPrepare_r(context, g1);
  if (prepared_geom_tmp == NULL) {
    return 2;
  }
  ret = GEOSPreparedContainsProperly_r(context, prepared_geom_tmp, g2);
  GEOSPreparedGeom_destroy_r(context, prepared_geom_tmp);
  return ret;
}
static void* contains_properly_func_tuple[2] = {GEOSContainsProperly,
                                                GEOSPreparedContainsProperly_r};
static void* contains_properly_data[1] = {contains_properly_func_tuple};
static void* covered_by_func_tuple[2] = {GEOSCoveredBy_r, GEOSPreparedCoveredBy_r};
static void* covered_by_data[1] = {covered_by_func_tuple};
static void* covers_func_tuple[2] = {GEOSCovers_r, GEOSPreparedCovers_r};
static void* covers_data[1] = {covers_func_tuple};
static void* crosses_func_tuple[2] = {GEOSCrosses_r, GEOSPreparedCrosses_r};
static void* crosses_data[1] = {crosses_func_tuple};
static void* disjoint_func_tuple[2] = {GEOSDisjoint_r, GEOSPreparedDisjoint_r};
static void* disjoint_data[1] = {disjoint_func_tuple};
static void* intersects_func_tuple[2] = {GEOSIntersects_r, GEOSPreparedIntersects_r};
static void* intersects_data[1] = {intersects_func_tuple};
static void* overlaps_func_tuple[2] = {GEOSOverlaps_r, GEOSPreparedOverlaps_r};
static void* overlaps_data[1] = {overlaps_func_tuple};
static void* touches_func_tuple[2] = {GEOSTouches_r, GEOSPreparedTouches_r};
static void* touches_data[1] = {touches_func_tuple};
static void* within_func_tuple[2] = {GEOSWithin_r, GEOSPreparedWithin_r};
static void* within_data[1] = {within_func_tuple};
static char YY_b_p_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_BOOL};
static void YY_b_p_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_YY_b* func = ((FuncGEOS_YY_b**)data)[0];
  FuncGEOS_YY_b* func_prepared = ((FuncGEOS_YY_b**)data)[1];

  GEOSGeometry *in1 = NULL, *in2 = NULL;
  GEOSPreparedGeometry* in1_prepared = NULL;
  char ret;

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom_with_prepared(*(GeometryObject**)ip1, &in1, &in1_prepared)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if ((in1 == NULL) || (in2 == NULL)) {
      /* in case of a missing value: return 0 (False) */
      ret = 0;
    } else {
      if (in1_prepared == NULL) {
        /* call the GEOS function */
        ret = func(ctx, in1, in2);
      } else {
        /* call the prepared GEOS function */
        ret = func_prepared(ctx, in1_prepared, in2);
      }
      /* return for illegal values */
      if (ret == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:

  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction YY_b_p_funcs[1] = {&YY_b_p_func};

static char is_prepared_dtypes[2] = {NPY_OBJECT, NPY_BOOL};
static void is_prepared_func(char** args, npy_intp* dimensions, npy_intp* steps,
                             void* data) {
  GEOSGeometry* in1 = NULL;
  GEOSPreparedGeometry* in1_prepared = NULL;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    /* get the geometry: return on error */
    if (!get_geom_with_prepared(*(GeometryObject**)ip1, &in1, &in1_prepared)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      break;
    }
    *(npy_bool*)op1 = (in1_prepared != NULL);
  }

  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction is_prepared_funcs[1] = {&is_prepared_func};

/* Define the geom -> geom functions (Y_Y) */
static void* envelope_data[1] = {GEOSEnvelope_r};
static void* convex_hull_data[1] = {GEOSConvexHull_r};
static void* GEOSBoundaryAllTypes_r(void* context, void* geom) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if (typ == 7) {
    /* return None for geometrycollections */
    return NULL;
  } else {
    return GEOSBoundary_r(context, geom);
  }
}
static void* boundary_data[1] = {GEOSBoundaryAllTypes_r};
static void* unary_union_data[1] = {GEOSUnaryUnion_r};
static void* point_on_surface_data[1] = {GEOSPointOnSurface_r};
static void* centroid_data[1] = {GEOSGetCentroid_r};
static void* line_merge_data[1] = {GEOSLineMerge_r};
static void* extract_unique_points_data[1] = {GEOSGeom_extractUniquePoints_r};
static void* GetExteriorRing(void* context, void* geom) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if (typ != 3) {
    return NULL;
  }
  void* ret = (void*)GEOSGetExteriorRing_r(context, geom);
  /* Create a copy of the obtained geometry */
  if (ret != NULL) {
    ret = GEOSGeom_clone_r(context, ret);
  }
  return ret;
}
static void* get_exterior_ring_data[1] = {GetExteriorRing};
/* the normalize funcion acts inplace */
static void* GEOSNormalize_r_with_clone(void* context, void* geom) {
  int ret;
  void* new_geom = GEOSGeom_clone_r(context, geom);
  if (new_geom == NULL) {
    return NULL;
  }
  ret = GEOSNormalize_r(context, new_geom);
  if (ret == -1) {
    GEOSGeom_destroy_r(context, new_geom);
    return NULL;
  }
  return new_geom;
}
static void* normalize_data[1] = {GEOSNormalize_r_with_clone};
static void* force_2d_data[1] = {PyGEOSForce2D};
#if GEOS_SINCE_3_8_0
static void* build_area_data[1] = {GEOSBuildArea_r};
static void* make_valid_data[1] = {GEOSMakeValid_r};
static void* coverage_union_data[1] = {GEOSCoverageUnion_r};
static void* GEOSMinimumBoundingCircleWithReturn(void* context, void* geom) {
  GEOSGeometry* center = NULL;
  double radius;
  GEOSGeometry* ret = GEOSMinimumBoundingCircle_r(context, geom, &radius, &center);
  if (ret == NULL) {
    return NULL;
  }
  GEOSGeom_destroy_r(context, center);
  return ret;
}
static void* minimum_bounding_circle_data[1] = {GEOSMinimumBoundingCircleWithReturn};
#endif
#if GEOS_SINCE_3_7_0
static void* reverse_data[1] = {GEOSReverse_r};
#endif
#if GEOS_SINCE_3_6_0
static void* oriented_envelope_data[1] = {GEOSMinimumRotatedRectangle_r};
#endif
typedef void* FuncGEOS_Y_Y(void* context, void* a);
static char Y_Y_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void Y_Y_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_Y_Y* func = (FuncGEOS_Y_Y*)data;
  GEOSGeometry* in1 = NULL;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(1);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    // get the geometry: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    if (in1 == NULL) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = func(ctx, in1);
      // NULL means: exception, but for some functions it may also indicate a
      // "missing value" (None) (GetExteriorRing, GEOSBoundaryAllTypes_r)
      // So: check the last_error before setting error state
      if ((geom_arr[i] == NULL) && (last_error[0] != 0)) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[1], steps[1], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction Y_Y_funcs[1] = {&Y_Y_func};

/* Define the geom -> no return value functions (Y) */
static char PrepareGeometryObject(void* ctx, GeometryObject* geom) {
  if (geom->ptr_prepared == NULL) {
    geom->ptr_prepared = (GEOSPreparedGeometry*)GEOSPrepare_r(ctx, geom->ptr);
    if (geom->ptr_prepared == NULL) {
      return PGERR_GEOS_EXCEPTION;
    }
  }
  return PGERR_SUCCESS;
}
static char DestroyPreparedGeometryObject(void* ctx, GeometryObject* geom) {
  if (geom->ptr_prepared != NULL) {
    GEOSPreparedGeom_destroy_r(ctx, geom->ptr_prepared);
    geom->ptr_prepared = NULL;
  }
  return PGERR_SUCCESS;
}

static void* prepare_data[1] = {PrepareGeometryObject};
static void* destroy_prepared_data[1] = {DestroyPreparedGeometryObject};
typedef char FuncPyGEOS_Y(void* ctx, GeometryObject* geom);
static char Y_dtypes[1] = {NPY_OBJECT};
static void Y_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncPyGEOS_Y* func = (FuncPyGEOS_Y*)data;
  GEOSGeometry* in1 = NULL;
  GeometryObject* geom_obj = NULL;

  GEOS_INIT;

  NO_OUTPUT_LOOP {
    geom_obj = *(GeometryObject**)ip1;
    if (!get_geom(geom_obj, &in1)) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }
    if (in1 != NULL) {
      errstate = func(ctx, geom_obj);
      if (errstate != PGERR_SUCCESS) {
        goto finish;
      }
    }
  }

finish:

  GEOS_FINISH;
}
static PyUFuncGenericFunction Y_funcs[1] = {&Y_func};

/* Define the geom, double -> geom functions (Yd_Y) */
static void* GEOSInterpolateProtectEmpty_r(void* context, void* geom, double d) {
  char errstate = geos_interpolate_checker(context, geom);
  if (errstate == PGERR_SUCCESS) {
    return GEOSInterpolate_r(context, geom, d);
  } else if (errstate == PGERR_EMPTY_GEOMETRY) {
    return GEOSGeom_createEmptyPoint_r(context);
  } else {
    return NULL;
  }
}
static void* line_interpolate_point_data[1] = {GEOSInterpolateProtectEmpty_r};
static void* GEOSInterpolateNormalizedProtectEmpty_r(void* context, void* geom,
                                                     double d) {
  char errstate = geos_interpolate_checker(context, geom);
  if (errstate == PGERR_SUCCESS) {
    return GEOSInterpolateNormalized_r(context, geom, d);
  } else if (errstate == PGERR_EMPTY_GEOMETRY) {
    return GEOSGeom_createEmptyPoint_r(context);
  } else {
    return NULL;
  }
}
static void* line_interpolate_point_normalized_data[1] = {
    GEOSInterpolateNormalizedProtectEmpty_r};

static void* simplify_data[1] = {GEOSSimplify_r};
static void* simplify_preserve_topology_data[1] = {GEOSTopologyPreserveSimplify_r};
static void* force_3d_data[1] = {PyGEOSForce3D};

#if GEOS_SINCE_3_9_0
static void* unary_union_prec_data[1] = {GEOSUnaryUnionPrec_r};
#endif

#if GEOS_SINCE_3_10_0
static void* segmentize_data[1] = {GEOSDensify_r};
#endif

typedef void* FuncGEOS_Yd_Y(void* context, void* a, double b);
static char Yd_Y_dtypes[3] = {NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};
static void Yd_Y_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_Yd_Y* func = (FuncGEOS_Yd_Y*)data;
  GEOSGeometry* in1 = NULL;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(2);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    // get the geometry: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    double in2 = *(double*)ip2;
    if ((in1 == NULL) || (npy_isnan(in2))) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = func(ctx, in1, in2);
      if (geom_arr[i] == NULL) {
        // Interpolate functions return NULL on PGERR_GEOMETRY_TYPE and on
        // PGERR_GEOS_EXCEPTION. Distinguish these by the state of last_error.
        errstate = last_error[0] == 0 ? PGERR_GEOMETRY_TYPE : PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction Yd_Y_funcs[1] = {&Yd_Y_func};

/* Define the geom, int -> geom functions (Yi_Y) */
/* We add bound and type checking to the various indexing functions */
static void* GetPointN(void* context, void* geom, int n) {
  char typ = GEOSGeomTypeId_r(context, geom);
  int size, i;
  if ((typ != 1) && (typ != 2)) {
    return NULL;
  }
  size = GEOSGeomGetNumPoints_r(context, geom);
  if (size == -1) {
    return NULL;
  }
  if (n < 0) {
    /* Negative indexing: we get it for free */
    i = size + n;
  } else {
    i = n;
  }
  if ((i < 0) || (i >= size)) {
    /* Important, could give segfaults else */
    return NULL;
  }
  return GEOSGeomGetPointN_r(context, geom, i);
}
static void* get_point_data[1] = {GetPointN};
static void* GetInteriorRingN(void* context, void* geom, int n) {
  char typ = GEOSGeomTypeId_r(context, geom);
  int size, i;
  if (typ != 3) {
    return NULL;
  }
  size = GEOSGetNumInteriorRings_r(context, geom);
  if (size == -1) {
    return NULL;
  }
  if (n < 0) {
    /* Negative indexing: we get it for free */
    i = size + n;
  } else {
    i = n;
  }
  if ((i < 0) || (i >= size)) {
    /* Important, could give segfaults else */
    return NULL;
  }
  void* ret = (void*)GEOSGetInteriorRingN_r(context, geom, i);
  /* Create a copy of the obtained geometry */
  if (ret != NULL) {
    ret = GEOSGeom_clone_r(context, ret);
  }
  return ret;
}
static void* get_interior_ring_data[1] = {GetInteriorRingN};
static void* GetGeometryN(void* context, void* geom, int n) {
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
  if ((i < 0) || (i >= size)) {
    /* Important, could give segfaults else */
    return NULL;
  }
  void* ret = (void*)GEOSGetGeometryN_r(context, geom, i);
  /* Create a copy of the obtained geometry */
  if (ret != NULL) {
    ret = GEOSGeom_clone_r(context, ret);
  }
  return ret;
}
static void* get_geometry_data[1] = {GetGeometryN};
/* the set srid funcion acts inplace */
static void* GEOSSetSRID_r_with_clone(void* context, void* geom, int srid) {
  void* ret = GEOSGeom_clone_r(context, geom);
  if (ret == NULL) {
    return NULL;
  }
  GEOSSetSRID_r(context, ret, srid);
  return ret;
}
static void* set_srid_data[1] = {GEOSSetSRID_r_with_clone};
typedef void* FuncGEOS_Yi_Y(void* context, void* a, int b);
static char Yi_Y_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void Yi_Y_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_Yi_Y* func = (FuncGEOS_Yi_Y*)data;
  GEOSGeometry* in1 = NULL;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(2);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    // get the geometry: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    int in2 = *(int*)ip2;
    if (in1 == NULL) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = func(ctx, in1, in2);
      // NULL means: exception, but for some functions it may also indicate a
      // "missing value" (None) (GetPointN, GetInteriorRingN, GetGeometryN)
      // So: check the last_error before setting error state
      if ((geom_arr[i] == NULL) && (last_error[0] != 0)) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction Yi_Y_funcs[1] = {&Yi_Y_func};

/* Define the geom, geom -> geom functions (YY_Y) */
static void* intersection_data[1] = {GEOSIntersection_r};
static void* difference_data[1] = {GEOSDifference_r};
static void* symmetric_difference_data[1] = {GEOSSymDifference_r};
static void* union_data[1] = {GEOSUnion_r};
static void* shared_paths_data[1] = {GEOSSharedPaths_r};
typedef void* FuncGEOS_YY_Y(void* context, void* a, void* b);
static char YY_Y_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};

/* There are two inner loop functions for the YY_Y. This is because this
 * pattern allows for ufunc.reduce.
 * A reduce operation requires special attention, because we output the
 * result in a temporary array. NumPy expects that the output array is
 * filled during the inner loop, so that it can take the first input of
 * the reduction operation to be the output array. Easiest to show by
 * example:

 * function called:         out = intersection.reduce(in)
 * initialization by numpy: out[0] = in[0]; in1 = out; in2[:] = in[:]
 * first loop:              out[0] = func(in1[0], in2[1])  [ = func(out[0], in2[1]) ]
 * second loop:             out[0] = func(in1[0], in2[2])  [ = func(out[0], in2[2]) ]
 */
static void YY_Y_func_reduce(char** args, npy_intp* dimensions, npy_intp* steps,
                             void* data) {
  FuncGEOS_YY_Y* func = (FuncGEOS_YY_Y*)data;
  GEOSGeometry *in1 = NULL, *in2 = NULL, *out = NULL;

  // Whether to destroy a temporary intermediate value of `out`:
  char do_destroy = 0;

  GEOS_INIT_THREADS;

  if (!get_geom(*(GeometryObject**)args[0], &out)) {
    errstate = PGERR_NOT_A_GEOMETRY;
  } else {
    BINARY_LOOP {
      // Get the geometry inputs; in1 from previous iteration, in2 from array
      in1 = out;
      if (!get_geom(*(GeometryObject**)ip2, &in2)) {
        errstate = PGERR_NOT_A_GEOMETRY;
        break;
      }

      /* Either (or both) in1 and in2 could be NULL (Python: None).
       * Reduction operations should skip None values. We have 4 possible combinations:
       */

      // 1. (not NULL, not NULL); run the GEOS function
      if ((in1 != NULL) && (in2 != NULL)) {
        out = func(ctx, in1, in2);

        // Discard in1 if it was a temporary intermediate
        if (do_destroy) {
          GEOSGeom_destroy_r(ctx, in1);
        }

        // Mark the newly generated geometry as intermediate. Note: out will become in1.
        do_destroy = 1;

        // Break on error (we do this after discarding in1 to avoid memleaks)
        if (out == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          break;
        }
      }

      // 2. (NULL, not NULL); When the first element of the reduction axis is None
      else if ((in1 == NULL) && (in2 != NULL)) {
        // Keep in2 as 'outcome' of the operation.
        out = in2;
        // Ensure that it will not be destroyed (it is owned by python)
        do_destroy = 0;
      }

      // 3. (not NULL, NULL); When a None value is encountered after a not-None
      //    Don't do `out = in1`, as that is already the case.
      // 4. (NULL, NULL); When we have not yet encountered any not-None
      //    Do nothing; out will remain NULL
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with a single PyObject while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(&out, args[2], steps[2], 1);
  }
}

static void YY_Y_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  // A reduce is characterized by multiple iterations (dimension[0] > 1) that
  // are output on the same place in memory (steps[2] == 0).
  if ((steps[2] == 0) && (dimensions[0] > 1)) {
    if (args[0] == args[2]) {
      YY_Y_func_reduce(args, dimensions, steps, data);
      return;
    } else {
      // Fail if inputs do not have the expected structure.
      PyErr_Format(PyExc_NotImplementedError,
                   "Unknown ufunc mode with args=[%p, %p, %p], steps=[%ld, %ld, %ld], "
                   "dimensions=[%ld].",
                   args[0], args[1], args[2], steps[0], steps[1], steps[2],
                   dimensions[0]);
      return;
    }
  }

  FuncGEOS_YY_Y* func = (FuncGEOS_YY_Y*)data;
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  GEOSGeometry** geom_arr;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    // get the geometries: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1) ||
        !get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    if ((in1 == NULL) || (in2 == NULL)) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = func(ctx, in1, in2);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction YY_Y_funcs[1] = {&YY_Y_func};

/* Define the geom -> double functions (Y_d) */
static int GetX(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetX_r(context, a, b);
  }
}
static void* get_x_data[1] = {GetX};
static int GetY(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetY_r(context, a, b);
  }
}
static void* get_y_data[1] = {GetY};
#if GEOS_SINCE_3_7_0
static int GetZ(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetZ_r(context, a, b);
  }
}
static void* get_z_data[1] = {GetZ};
#endif
static void* area_data[1] = {GEOSArea_r};
static void* length_data[1] = {GEOSLength_r};

#if GEOS_SINCE_3_6_0
static int GetPrecision(void* context, void* a, double* b) {
  // GEOS returns -1 on error; 0 indicates double precision; > 0 indicates a precision
  // grid size was set for this geometry.
  double out = GEOSGeom_getPrecision_r(context, a);
  if (out == -1) {
    return 0;
  }
  *(double*)b = out;
  return 1;
}
static void* get_precision_data[1] = {GetPrecision};
static int MinimumClearance(void* context, void* a, double* b) {
  // GEOSMinimumClearance deviates from the pattern of returning 0 on exception and 1 on
  // success for functions that return an int (it follows pattern for boolean functions
  // returning char 0/1 and 2 on exception)
  int retcode = GEOSMinimumClearance_r(context, a, b);
  if (retcode == 2) {
    return 0;
  } else {
    return 1;
  }
}
static void* minimum_clearance_data[1] = {MinimumClearance};
#endif
#if GEOS_SINCE_3_8_0
static int GEOSMinimumBoundingRadius(void* context, GEOSGeometry* geom, double* radius) {
  GEOSGeometry* center = NULL;
  GEOSGeometry* ret = GEOSMinimumBoundingCircle_r(context, geom, radius, &center);
  if (ret == NULL) {
    return 0;  // exception code
  }
  GEOSGeom_destroy_r(context, center);
  GEOSGeom_destroy_r(context, ret);
  return 1;  // success code
}
static void* minimum_bounding_radius_data[1] = {GEOSMinimumBoundingRadius};
#endif
typedef int FuncGEOS_Y_d(void* context, void* a, double* b);
static char Y_d_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};
static void Y_d_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_Y_d* func = (FuncGEOS_Y_d*)data;
  GEOSGeometry* in1 = NULL;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    /* get the geometry: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (in1 == NULL) {
      *(double*)op1 = NPY_NAN;
    } else {
      /* let the GEOS function set op1; return on error */
      if (func(ctx, in1, (npy_double*)op1) == 0) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction Y_d_funcs[1] = {&Y_d_func};

/* Define the geom -> int functions (Y_i) */
/* data values are GEOS func, GEOS error code, return value when input is None */
static void* get_type_id_func_tuple[3] = {GEOSGeomTypeId_r, (void*)-1, (void*)-1};
static void* get_type_id_data[1] = {get_type_id_func_tuple};

static void* get_dimensions_func_tuple[3] = {GEOSGeom_getDimensions_r, (void*)0,
                                             (void*)-1};
static void* get_dimensions_data[1] = {get_dimensions_func_tuple};

static void* get_coordinate_dimension_func_tuple[3] = {GEOSGeom_getCoordinateDimension_r,
                                                       (void*)-1, (void*)-1};
static void* get_coordinate_dimension_data[1] = {get_coordinate_dimension_func_tuple};

static void* get_srid_func_tuple[3] = {GEOSGetSRID_r, (void*)0, (void*)-1};
static void* get_srid_data[1] = {get_srid_func_tuple};

static int GetNumPoints(void* context, void* geom, int n) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if ((typ == 1) || (typ == 2)) { /* Linestring & Linearring */
    return GEOSGeomGetNumPoints_r(context, geom);
  } else {
    return 0;
  }
}
static void* get_num_points_func_tuple[3] = {GetNumPoints, (void*)-1, (void*)0};
static void* get_num_points_data[1] = {get_num_points_func_tuple};

static int GetNumInteriorRings(void* context, void* geom, int n) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if (typ == 3) { /* Polygon */
    return GEOSGetNumInteriorRings_r(context, geom);
  } else {
    return 0;
  }
}
static void* get_num_interior_rings_func_tuple[3] = {GetNumInteriorRings, (void*)-1,
                                                     (void*)0};
static void* get_num_interior_rings_data[1] = {get_num_interior_rings_func_tuple};

static void* get_num_geometries_func_tuple[3] = {GEOSGetNumGeometries_r, (void*)-1,
                                                 (void*)0};
static void* get_num_geometries_data[1] = {get_num_geometries_func_tuple};

static void* get_num_coordinates_func_tuple[3] = {GEOSGetNumCoordinates_r, (void*)-1,
                                                  (void*)0};
static void* get_num_coordinates_data[1] = {get_num_coordinates_func_tuple};

typedef int FuncGEOS_Y_i(void* context, void* a);
static char Y_i_dtypes[2] = {NPY_OBJECT, NPY_INT};
static void Y_i_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_Y_i* func = ((FuncGEOS_Y_i**)data)[0];
  int errcode = (int)((int**)data)[1];
  int none_value = (int)((int**)data)[2];

  GEOSGeometry* in1 = NULL;
  int result;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    /* get the geometry: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (in1 == NULL) {
      /* None results in 0 for counting functions, -1 otherwise */
      *(npy_int*)op1 = none_value;
    } else {
      result = func(ctx, in1);
      // Check last_error if the result equals errcode.
      // Otherwise we can't be sure if it is an exception
      if ((result == errcode) && (last_error[0] != 0)) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      *(npy_int*)op1 = result;
    }
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction Y_i_funcs[1] = {&Y_i_func};

/* Define the geom, geom -> double functions (YY_d) */
static void* distance_data[1] = {GEOSDistance_r};
static void* hausdorff_distance_data[1] = {GEOSHausdorffDistance_r};
#if GEOS_SINCE_3_7_0
static int GEOSFrechetDistanceWrapped_r(void* context, void* a, void* b, double* c) {
  /* Handle empty geometries (they give segfaults) */
  if (GEOSisEmpty_r(context, a) || GEOSisEmpty_r(context, b)) {
    *c = NPY_NAN;
    return 1;
  }
  return GEOSFrechetDistance_r(context, a, b, c);
}
static void* frechet_distance_data[1] = {GEOSFrechetDistanceWrapped_r};
#endif
/* Project and ProjectNormalize don't return error codes. wrap them. */
static int GEOSProjectWrapped_r(void* context, void* a, void* b, double* c) {
  /* Handle empty points (they give segfaults (for b) or give exception (for a)) */
  if (GEOSisEmpty_r(context, a) || GEOSisEmpty_r(context, b)) {
    *c = NPY_NAN;
  } else {
    *c = GEOSProject_r(context, a, b);
  }
  if (*c == -1.0) {
    return 0;
  } else {
    return 1;
  }
}
static void* line_locate_point_data[1] = {GEOSProjectWrapped_r};
static int GEOSProjectNormalizedWrapped_r(void* context, void* a, void* b, double* c) {
  double length;
  double distance;

  /* Handle empty points (they give segfaults (for b) or give exception (for a)) */
  if (GEOSisEmpty_r(context, a) || GEOSisEmpty_r(context, b)) {
    *c = NPY_NAN;
  } else {
    /* Use custom implementation of GEOSProjectNormalized to overcome bug in
    older GEOS versions (https://trac.osgeo.org/geos/ticket/1058) */
    if (GEOSLength_r(context, a, &length) != 1) {
      return 0;
    };
    distance = GEOSProject_r(context, a, b);
    if (distance == -1.0) {
      return 0;
    } else {
      *c = distance / length;
    }
  }
  return 1;
}
static void* line_locate_point_normalized_data[1] = {GEOSProjectNormalizedWrapped_r};
typedef int FuncGEOS_YY_d(void* context, void* a, void* b, double* c);
static char YY_d_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE};
static void YY_d_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_YY_d* func = (FuncGEOS_YY_d*)data;
  GEOSGeometry *in1 = NULL, *in2 = NULL;

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if ((in1 == NULL) || (in2 == NULL)) {
      /* in case of a missing value: return NaN */
      *(double*)op1 = NPY_NAN;
    } else {
      /* let the GEOS function set op1; return on error */
      if (func(ctx, in1, in2, (double*)op1) == 0) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      /* incase the outcome is 0.0, check the inputs for emptyness */
      if (*op1 == 0.0) {
        if (GEOSisEmpty_r(ctx, in1) || GEOSisEmpty_r(ctx, in2)) {
          *(double*)op1 = NPY_NAN;
        }
      }
    }
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction YY_d_funcs[1] = {&YY_d_func};

/* Define the geom, geom, double -> double functions (YYd_d) */
static void* hausdorff_distance_densify_data[1] = {GEOSHausdorffDistanceDensify_r};
#if GEOS_SINCE_3_7_0
static void* frechet_distance_densify_data[1] = {GEOSFrechetDistanceDensify_r};
#endif
typedef int FuncGEOS_YYd_d(void* context, void* a, void* b, double c, double* d);
static char YYd_d_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE};
static void YYd_d_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_YYd_d* func = (FuncGEOS_YYd_d*)data;
  GEOSGeometry *in1 = NULL, *in2 = NULL;

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    double in3 = *(double*)ip3;
    if ((in1 == NULL) || (in2 == NULL) || npy_isnan(in3) || GEOSisEmpty_r(ctx, in1) ||
        GEOSisEmpty_r(ctx, in2)) {
      *(double*)op1 = NPY_NAN;
    } else {
      /* let the GEOS function set op1; return on error */
      if (func(ctx, in1, in2, in3, (double*)op1) == 0) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction YYd_d_funcs[1] = {&YYd_d_func};

#if GEOS_SINCE_3_9_0

/* Define the geom, geom, double -> geom functions (YYd_Y) */
static void* intersection_prec_data[1] = {GEOSIntersectionPrec_r};
static void* difference_prec_data[1] = {GEOSDifferencePrec_r};
static void* symmetric_difference_prec_data[1] = {GEOSSymDifferencePrec_r};
static void* union_prec_data[1] = {GEOSUnionPrec_r};
typedef void* FuncGEOS_YYd_Y(void* context, void* a, void* b, double c);
static char YYd_Y_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};

static void YYd_Y_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  FuncGEOS_YYd_Y* func = (FuncGEOS_YYd_Y*)data;
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  GEOSGeometry** geom_arr;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    // get the geometries: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1) ||
        !get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    double in3 = *(double*)ip3;
    if ((in1 == NULL) || (in2 == NULL) || npy_isnan(in3)) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = func(ctx, in1, in2, in3);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[3], steps[3], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction YYd_Y_funcs[1] = {&YYd_Y_func};
#endif

/* Define functions with unique call signatures */
static char box_dtypes[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                             NPY_DOUBLE, NPY_BOOL,   NPY_OBJECT};
static void box_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4];
  npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4];
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(5);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * n);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  for (i = 0; i < n; i++, ip1 += is1, ip2 += is2, ip3 += is3, ip4 += is4, ip5 += is5) {
    geom_arr[i] = create_box(ctx, *(double*)ip1, *(double*)ip2, *(double*)ip3,
                             *(double*)ip4, *(char*)ip5);
    if (geom_arr[i] == NULL) {
      // result will be NULL for any nan coordinates, which is OK;
      // otherwise raise an error
      if (!(npy_isnan(*(double*)ip1) || npy_isnan(*(double*)ip2) ||
            npy_isnan(*(double*)ip3) || npy_isnan(*(double*)ip4))) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[5], steps[5], dimensions[0]);
  }
  free(geom_arr);
}

static PyUFuncGenericFunction box_funcs[1] = {&box_func};

static void* null_data[1] = {NULL};
static char buffer_inner(void* ctx, GEOSBufferParams* params, void* ip1, void* ip2,
                         GEOSGeometry** geom_arr, npy_intp i) {
  GEOSGeometry* in1 = NULL;

  /* get the geometry: return on error */
  if (!get_geom(*(GeometryObject**)ip1, &in1)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  double in2 = *(double*)ip2;
  /* handle NULL geometries or NaN buffer width */
  if ((in1 == NULL) || npy_isnan(in2)) {
    geom_arr[i] = NULL;
  } else {
    geom_arr[i] = GEOSBufferWithParams_r(ctx, in1, params, in2);
    if (geom_arr[i] == NULL) {
      return PGERR_GEOS_EXCEPTION;
    }
  }
  return PGERR_SUCCESS;
}

static char buffer_dtypes[8] = {NPY_OBJECT, NPY_DOUBLE, NPY_INT,  NPY_INT,
                                NPY_INT,    NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void buffer_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4],
       *ip6 = args[5], *ip7 = args[6];
  npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4],
           is6 = steps[5], is7 = steps[6];
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(7);

  if ((is3 != 0) || (is4 != 0) || (is5 != 0) || (is6 != 0) || (is7 != 0)) {
    PyErr_Format(PyExc_ValueError, "Buffer function called with non-scalar parameters");
    return;
  }

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * n);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  GEOSBufferParams* params = GEOSBufferParams_create_r(ctx);
  if (params != 0) {
    if (!GEOSBufferParams_setQuadrantSegments_r(ctx, params, *(int*)ip3)) {
      errstate = PGERR_GEOS_EXCEPTION;
    }
    if (!GEOSBufferParams_setEndCapStyle_r(ctx, params, *(int*)ip4)) {
      errstate = PGERR_GEOS_EXCEPTION;
    }
    if (!GEOSBufferParams_setJoinStyle_r(ctx, params, *(int*)ip5)) {
      errstate = PGERR_GEOS_EXCEPTION;
    }
    if (!GEOSBufferParams_setMitreLimit_r(ctx, params, *(double*)ip6)) {
      errstate = PGERR_GEOS_EXCEPTION;
    }
    if (!GEOSBufferParams_setSingleSided_r(ctx, params, *(npy_bool*)ip7)) {
      errstate = PGERR_GEOS_EXCEPTION;
    }
  } else {
    errstate = PGERR_GEOS_EXCEPTION;
  }

  if (errstate == PGERR_SUCCESS) {
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
      errstate = buffer_inner(ctx, params, ip1, ip2, geom_arr, i);
      if (errstate != PGERR_SUCCESS) {
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  if (params != 0) {
    GEOSBufferParams_destroy_r(ctx, params);
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[7], steps[7], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction buffer_funcs[1] = {&buffer_func};

static char offset_curve_dtypes[6] = {NPY_OBJECT, NPY_DOUBLE, NPY_INT,
                                      NPY_INT,    NPY_DOUBLE, NPY_OBJECT};
static void offset_curve_func(char** args, npy_intp* dimensions, npy_intp* steps,
                              void* data) {
  char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4];
  npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4];
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSGeometry** geom_arr;
  GEOSGeometry* in1 = NULL;

  CHECK_NO_INPLACE_OUTPUT(5);

  if ((is3 != 0) || (is4 != 0) || (is5 != 0)) {
    PyErr_Format(PyExc_ValueError,
                 "Offset curve function called with non-scalar parameters");
    return;
  }

  double width;
  int quadsegs = *(int*)ip3;
  int joinStyle = *(int*)ip4;
  double mitreLimit = *(double*)ip5;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * n);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
    /* get the geometry: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }

    width = *(double*)ip2;
    if ((in1 == NULL) || npy_isnan(width)) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = GEOSOffsetCurve_r(ctx, in1, width, quadsegs, joinStyle, mitreLimit);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[5], steps[5], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction offset_curve_funcs[1] = {&offset_curve_func};

static char snap_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};
static void snap_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(3);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1) ||
        !get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    double in3 = *(double*)ip3;
    if ((in1 == NULL) || (in2 == NULL) || npy_isnan(in3)) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = GEOSSnap_r(ctx, in1, in2, in3);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[3], steps[3], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction snap_funcs[1] = {&snap_func};

static char clip_by_rect_dtypes[6] = {NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE,
                                      NPY_DOUBLE, NPY_DOUBLE, NPY_OBJECT};
static void clip_by_rect_func(char** args, npy_intp* dimensions, npy_intp* steps,
                              void* data) {
  char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4];
  npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4];
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSGeometry** geom_arr;
  GEOSGeometry* in1 = NULL;

  CHECK_NO_INPLACE_OUTPUT(5);

  if ((is2 != 0) || (is3 != 0) || (is4 != 0) || (is5 != 0)) {
    PyErr_Format(PyExc_ValueError,
                 "clip_by_rect function called with non-scalar parameters");
    return;
  }

  double xmin = *(double*)ip2;
  double ymin = *(double*)ip3;
  double xmax = *(double*)ip4;
  double ymax = *(double*)ip5;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * n);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  for (i = 0; i < n; i++, ip1 += is1) {
    /* get the geometry: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }

    if (in1 == NULL) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = GEOSClipByRect_r(ctx, in1, xmin, ymin, xmax, ymax);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[5], steps[5], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction clip_by_rect_funcs[1] = {&clip_by_rect_func};

static char equals_exact_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_BOOL};
static void equals_exact_func(char** args, npy_intp* dimensions, npy_intp* steps,
                              void* data) {
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  double in3;
  npy_bool ret;

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    in3 = *(double*)ip3;
    if ((in1 == NULL) || (in2 == NULL) || npy_isnan(in3)) {
      /* return 0 (False) for missing values */
      ret = 0;
    } else {
      ret = GEOSEqualsExact_r(ctx, in1, in2, in3);
      if ((ret != 0) && (ret != 1)) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction equals_exact_funcs[1] = {&equals_exact_func};

#if GEOS_SINCE_3_10_0

static char dwithin_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_BOOL};
static void dwithin_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  GEOSPreparedGeometry* in1_prepared = NULL;
  double in3;
  npy_bool ret;

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom_with_prepared(*(GeometryObject**)ip1, &in1, &in1_prepared)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    in3 = *(double*)ip3;
    if ((in1 == NULL) || (in2 == NULL) || npy_isnan(in3)) {
      /* in case of a missing value: return 0 (False) */
      ret = 0;
    } else {
      if (in1_prepared == NULL) {
        /* call the GEOS function */
        ret = GEOSDistanceWithin_r(ctx, in1, in2, in3);
      } else {
        /* call the prepared GEOS function */
        ret = GEOSPreparedDistanceWithin_r(ctx, in1_prepared, in2, in3);
      }
      /* return for illegal values */
      if (ret == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:

  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction dwithin_funcs[1] = {&dwithin_func};

#endif  // GEOS_SINCE_3_10_0

static char delaunay_triangles_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void delaunay_triangles_func(char** args, npy_intp* dimensions, npy_intp* steps,
                                    void* data) {
  GEOSGeometry* in1 = NULL;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(3);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    // get the geometry: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    double in2 = *(double*)ip2;
    npy_bool in3 = *(npy_bool*)ip3;
    if ((in1 == NULL) || npy_isnan(in2)) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = GEOSDelaunayTriangulation_r(ctx, in1, in2, (int)in3);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[3], steps[3], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction delaunay_triangles_funcs[1] = {&delaunay_triangles_func};

static char voronoi_polygons_dtypes[5] = {NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT, NPY_BOOL,
                                          NPY_OBJECT};
static void voronoi_polygons_func(char** args, npy_intp* dimensions, npy_intp* steps,
                                  void* data) {
  GEOSGeometry *in1 = NULL, *in3 = NULL;
  GEOSGeometry** geom_arr;

  CHECK_NO_INPLACE_OUTPUT(4);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  QUATERNARY_LOOP {
    // get the geometry: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1) ||
        !get_geom(*(GeometryObject**)ip3, &in3)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    double in2 = *(double*)ip2;
    npy_bool in4 = *(npy_bool*)ip4;
    if ((in1 == NULL) || npy_isnan(in2)) {
      /* propagate NULL geometries; in3 = NULL is actually supported */
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = GEOSVoronoiDiagram_r(ctx, in1, in3, in2, (int)in4);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[4], steps[4], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction voronoi_polygons_funcs[1] = {&voronoi_polygons_func};

static char is_valid_reason_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void is_valid_reason_func(char** args, npy_intp* dimensions, npy_intp* steps,
                                 void* data) {
  char* reason;
  GEOSGeometry* in1 = NULL;

  GEOS_INIT;

  UNARY_LOOP {
    PyObject** out = (PyObject**)op1;
    /* get the geometry return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (in1 == NULL) {
      /* Missing geometries give None */
      Py_XDECREF(*out);
      Py_INCREF(Py_None);
      *out = Py_None;
    } else {
      reason = GEOSisValidReason_r(ctx, in1);
      if (reason == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      /* convert to python string and set to out */
      Py_XDECREF(*out);
      *out = PyUnicode_FromString(reason);
      GEOSFree_r(ctx, reason);
    }
  }

finish:
  GEOS_FINISH;
}
static PyUFuncGenericFunction is_valid_reason_funcs[1] = {&is_valid_reason_func};

static char relate_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void relate_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  char* pattern;
  GEOSGeometry *in1 = NULL, *in2 = NULL;

  GEOS_INIT;

  BINARY_LOOP {
    PyObject** out = (PyObject**)op1;
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if ((in1 == NULL) || (in2 == NULL)) {
      /* Missing geometries give None */
      Py_XDECREF(*out);
      Py_INCREF(Py_None);
      *out = Py_None;
    } else {
      pattern = GEOSRelate_r(ctx, in1, in2);
      if (pattern == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      /* convert to python string and set to out */
      Py_XDECREF(*out);
      *out = PyUnicode_FromString(pattern);
      GEOSFree_r(ctx, pattern);
    }
  }

finish:
  GEOS_FINISH;
}
static PyUFuncGenericFunction relate_funcs[1] = {&relate_func};

static char relate_pattern_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT, NPY_BOOL};
static void relate_pattern_func(char** args, npy_intp* dimensions, npy_intp* steps,
                                void* data) {
  GEOSGeometry *in1 = NULL, *in2 = NULL;
  const char* pattern = NULL;
  npy_bool ret;

  /* get the pattern argument (only deal with scalar for now) */
  char* ip3 = args[2];
  npy_intp is3 = steps[2];

  if (is3 != 0) {
    PyErr_Format(PyExc_ValueError, "pattern keyword only supports scalar argument");
    return;
  }
  PyObject* in3 = *(PyObject**)ip3;
  if (PyUnicode_Check(in3)) {
    pattern = PyUnicode_AsUTF8(in3);
    if (pattern == NULL) {
      /* error happened in PyUnicode_AsUTF8, error already set by Python */
      return;
    }
  } else {
    PyErr_Format(PyExc_TypeError, "pattern keyword expected string, got %s",
                 Py_TYPE(in3)->tp_name);
    return;
  }

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    /* ip3 is already handled above */

    if ((in1 == NULL) || (in2 == NULL)) {
      /* in case of a missing value: return 0 (False) */
      ret = 0;
    } else {
      ret = GEOSRelatePattern_r(ctx, in1, in2, pattern);
      if (ret == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction relate_pattern_funcs[1] = {&relate_pattern_func};

static char polygonize_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void polygonize_func(char** args, npy_intp* dimensions, npy_intp* steps,
                            void* data) {
  GEOSGeometry* geom = NULL;
  unsigned int n_geoms;

  GEOS_INIT;

  GEOSGeometry** geoms = malloc(sizeof(void*) * dimensions[1]);
  if (geoms == NULL) {
    errstate = PGERR_NO_MALLOC;
    goto finish;
  }

  SINGLE_COREDIM_LOOP_OUTER {
    n_geoms = 0;
    SINGLE_COREDIM_LOOP_INNER {
      if (!get_geom(*(GeometryObject**)cp1, &geom)) {
        errstate = PGERR_NOT_A_GEOMETRY;
        goto finish;
      }
      if (geom == NULL) {
        continue;
      }
      geoms[n_geoms] = geom;
      n_geoms++;
    }

    GEOSGeometry* ret_ptr = GEOSPolygonize_r(ctx, geoms, n_geoms);
    if (ret_ptr == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }
    OUTPUT_Y;
  }

finish:
  if (geoms != NULL) {
    free(geoms);
  }
  GEOS_FINISH;
}
static PyUFuncGenericFunction polygonize_funcs[1] = {&polygonize_func};

static char polygonize_full_dtypes[5] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT, NPY_OBJECT,
                                         NPY_OBJECT};
static void polygonize_full_func(char** args, npy_intp* dimensions, npy_intp* steps,
                                 void* data) {
  GEOSGeometry* geom = NULL;
  GEOSGeometry* geom_copy = NULL;
  unsigned int n_geoms;

  GEOSGeometry* collection = NULL;
  GEOSGeometry* cuts = NULL;
  GEOSGeometry* dangles = NULL;
  GEOSGeometry* invalidRings = NULL;

  GEOS_INIT;

  GEOSGeometry** geoms = malloc(sizeof(void*) * dimensions[1]);
  if (geoms == NULL) {
    errstate = PGERR_NO_MALLOC;
    goto finish;
  }

  SINGLE_COREDIM_LOOP_OUTER_NOUT4 {
    n_geoms = 0;
    SINGLE_COREDIM_LOOP_INNER {
      if (!get_geom(*(GeometryObject**)cp1, &geom)) {
        errstate = PGERR_NOT_A_GEOMETRY;
        goto finish;
      }
      if (geom == NULL) {
        continue;
      }
      // need to copy the input geometries, because the Collection takes ownership
      geom_copy = GEOSGeom_clone_r(ctx, geom);
      if (geom_copy == NULL) {
        // if something went wrong before creating the collection, destroy previously
        // cloned geoms
        for (i = 0; i < n_geoms; i++) {
          GEOSGeom_destroy_r(ctx, geoms[i]);
        }
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      geoms[n_geoms] = geom_copy;
      n_geoms++;
    }
    collection =
        GEOSGeom_createCollection_r(ctx, GEOS_GEOMETRYCOLLECTION, geoms, n_geoms);
    if (collection == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }

    GEOSGeometry* ret_ptr =
        GEOSPolygonize_full_r(ctx, collection, &cuts, &dangles, &invalidRings);
    if (ret_ptr == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }
    OUTPUT_Y_I(1, ret_ptr);
    OUTPUT_Y_I(2, cuts);
    OUTPUT_Y_I(3, dangles);
    OUTPUT_Y_I(4, invalidRings);
    GEOSGeom_destroy_r(ctx, collection);
    collection = NULL;
  }

finish:
  if (collection != NULL) {
    GEOSGeom_destroy_r(ctx, collection);
  }
  if (geoms != NULL) {
    free(geoms);
  }
  GEOS_FINISH;
}
static PyUFuncGenericFunction polygonize_full_funcs[1] = {&polygonize_full_func};

static char shortest_line_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void shortest_line_func(char** args, npy_intp* dimensions, npy_intp* steps,
                               void* data) {
  GEOSGeometry* in1 = NULL;
  GEOSGeometry* in2 = NULL;
  GEOSPreparedGeometry* in1_prepared = NULL;
  GEOSGeometry** geom_arr;
  GEOSCoordSequence* coord_seq = NULL;

  CHECK_NO_INPLACE_OUTPUT(2);

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  BINARY_LOOP {
    /* get the geometries: return on error */
    if (!get_geom_with_prepared(*(GeometryObject**)ip1, &in1, &in1_prepared)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    if (!get_geom(*(GeometryObject**)ip2, &in2)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }

    if ((in1 == NULL) || (in2 == NULL) || GEOSisEmpty_r(ctx, in1) ||
        GEOSisEmpty_r(ctx, in2)) {
      // in case of a missing value or empty geometry: return NULL (None)
      // GEOSNearestPoints_r returns NULL for empty geometries
      // but this is not distinguishable from an actual error, so we handle this ourselves
      geom_arr[i] = NULL;
      continue;
    }
#if GEOS_SINCE_3_9_0
    if (in1_prepared != NULL) {
      coord_seq = GEOSPreparedNearestPoints_r(ctx, in1_prepared, in2);
    } else {
      coord_seq = GEOSNearestPoints_r(ctx, in1, in2);
    }
#else
    coord_seq = GEOSNearestPoints_r(ctx, in1, in2);
#endif
    if (coord_seq == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    geom_arr[i] = GEOSGeom_createLineString_r(ctx, coord_seq);
    // Note: coordinate sequence is owned by linestring; if linestring fails to
    // construct, it will automatically clean up the coordinate sequence
    if (geom_arr[i] == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction shortest_line_funcs[1] = {&shortest_line_func};

#if GEOS_SINCE_3_6_0
static char set_precision_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_INT, NPY_OBJECT};
static void set_precision_func(char** args, npy_intp* dimensions, npy_intp* steps,
                               void* data) {
  GEOSGeometry* in1 = NULL;
  GEOSGeometry** geom_arr;
  int flags;

  CHECK_NO_INPLACE_OUTPUT(3);

  /* preserve topology flag
   * flags:
   * - 0: default (from GEOS 3.10 this is named GEOS_PREC_VALID_OUTPUT)
   * - 1: GEOS_PREC_NO_TOPO
   * - 2: GEOS_PREC_KEEP_COLLAPSED
   */
  if (steps[2] != 0) {
    PyErr_Format(PyExc_ValueError, "set_precision function called with non-scalar mode");
    return;
  }
  flags = *(int*)args[2];
  if (!((flags == 0) || (flags == GEOS_PREC_NO_TOPO) ||
        (flags == GEOS_PREC_KEEP_COLLAPSED))) {
    PyErr_Format(PyExc_ValueError, "set_precision function called with illegal mode");
    return;
  }

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    // get the geometry: return on error
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    // grid size
    double in2 = *(double*)ip2;

    if ((in1 == NULL) || npy_isnan(in2)) {
      // in case of a missing value: return NULL (None)
      geom_arr[i] = NULL;
    } else {
      geom_arr[i] = GEOSGeom_setPrecision_r(ctx, in1, in2, flags);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      }
    }
  }

  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[3], steps[3], dimensions[0]);
  }
  free(geom_arr);
}

static PyUFuncGenericFunction set_precision_funcs[1] = {&set_precision_func};
#endif

/* define double -> geometry construction functions */
static char points_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void points_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  GEOSCoordSequence* coord_seq = NULL;
  GEOSGeometry** geom_arr;

  // check the ordinate dimension before calling GEOSCoordSeq_create_r
  if (dimensions[1] < 2 || dimensions[1] > 3) {
    PyErr_Format(PyExc_ValueError,
                 "The ordinate (last) dimension should be 2 or 3, got %ld",
                 dimensions[1]);
    return;
  }

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  SINGLE_COREDIM_LOOP_OUTER {
    coord_seq = GEOSCoordSeq_create_r(ctx, 1, n_c1);
    if (coord_seq == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    SINGLE_COREDIM_LOOP_INNER {
      if (!GEOSCoordSeq_setOrdinate_r(ctx, coord_seq, 0, i_c1, *(double*)cp1)) {
        errstate = PGERR_GEOS_EXCEPTION;
        GEOSCoordSeq_destroy_r(ctx, coord_seq);
        destroy_geom_arr(ctx, geom_arr, i - 1);
        goto finish;
      }
    }
    geom_arr[i] = GEOSGeom_createPoint_r(ctx, coord_seq);
    // Note: coordinate sequence is owned by point; if point fails to construct, it will
    // automatically clean up the coordinate sequence
    if (geom_arr[i] == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;
  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[1], steps[1], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction points_funcs[1] = {&points_func};

static char linestrings_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void linestrings_func(char** args, npy_intp* dimensions, npy_intp* steps,
                             void* data) {
  GEOSCoordSequence* coord_seq = NULL;
  GEOSGeometry** geom_arr;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  DOUBLE_COREDIM_LOOP_OUTER {
    coord_seq = coordseq_from_buffer(ctx, (double*)ip1, n_c1, n_c2, 0, cs1, cs2);
    if (coord_seq == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    geom_arr[i] = GEOSGeom_createLineString_r(ctx, coord_seq);
    // Note: coordinate sequence is owned by linestring; if linestring fails to construct,
    // it will automatically clean up the coordinate sequence
    if (geom_arr[i] == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[1], steps[1], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction linestrings_funcs[1] = {&linestrings_func};

static char linearrings_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void linearrings_func(char** args, npy_intp* dimensions, npy_intp* steps,
                             void* data) {
  GEOSCoordSequence* coord_seq = NULL;
  GEOSGeometry** geom_arr;
  char ring_closure = 0;
  double first_coord, last_coord;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr);

  GEOS_INIT_THREADS;

  DOUBLE_COREDIM_LOOP_OUTER {
    /* check if first and last coords are equal; duplicate if necessary */
    ring_closure = 0;
    if (n_c1 == 3) {
      ring_closure = 1;
    } else {
      DOUBLE_COREDIM_LOOP_INNER_2 {
        first_coord = *(double*)(ip1 + i_c2 * cs2);
        last_coord = *(double*)(ip1 + (n_c1 - 1) * cs1 + i_c2 * cs2);
        if (first_coord != last_coord) {
          ring_closure = 1;
          break;
        }
      }
    }
    /* the minimum number of coordinates in a linearring is 4 */
    if (n_c1 + ring_closure < 4) {
      errstate = PGERR_LINEARRING_NCOORDS;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    /* fill the coordinate sequence */
    coord_seq =
        coordseq_from_buffer(ctx, (double*)ip1, n_c1, n_c2, ring_closure, cs1, cs2);
    if (coord_seq == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    geom_arr[i] = GEOSGeom_createLinearRing_r(ctx, coord_seq);
    // Note: coordinate sequence is owned by linearring; if linearring fails to construct,
    // it will automatically clean up the coordinate sequence
    if (geom_arr[i] == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[1], steps[1], dimensions[0]);
  }
  free(geom_arr);
}
static PyUFuncGenericFunction linearrings_funcs[1] = {&linearrings_func};

static char polygons_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void polygons_func(char** args, npy_intp* dimensions, npy_intp* steps,
                          void* data) {
  GEOSGeometry *hole, *shell, *hole_copy, *shell_copy;
  GEOSGeometry **holes, **geom_arr;
  int geom_type;
  int n_holes;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr)

  // allocate a temporary array to store holes
  holes = malloc(sizeof(void*) * dimensions[1]);
  CHECK_ALLOC(holes)

  GEOS_INIT_THREADS;

  BINARY_SINGLE_COREDIM_LOOP_OUTER {
    if (!get_geom(*(GeometryObject**)ip1, &shell)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    if (shell == NULL) {
      // output empty polygon if shell is None (ignoring holes)
      geom_arr[i] = GEOSGeom_createEmptyPolygon_r(ctx);
      if (geom_arr[i] == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        break;
      };
      continue;
    }
    geom_type = GEOSGeomTypeId_r(ctx, shell);
    // Pre-emptively check the geometry type (https://trac.osgeo.org/geos/ticket/1111)
    if (geom_type == -1) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    } else if (geom_type != 2) {
      errstate = PGERR_GEOMETRY_TYPE;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    }
    n_holes = 0;
    cp1 = ip2;
    BINARY_SINGLE_COREDIM_LOOP_INNER {
      if (!get_geom(*(GeometryObject**)cp1, &hole)) {
        errstate = PGERR_NOT_A_GEOMETRY;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        destroy_geom_arr(ctx, holes, n_holes - 1);
        goto finish;
      }
      if (hole == NULL) {
        continue;
      }
      // Pre-emptively check the geometry type (https://trac.osgeo.org/geos/ticket/1111)
      geom_type = GEOSGeomTypeId_r(ctx, hole);
      if (geom_type == -1) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        destroy_geom_arr(ctx, holes, n_holes - 1);
        goto finish;
      } else if (geom_type != 2) {
        errstate = PGERR_GEOMETRY_TYPE;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        destroy_geom_arr(ctx, holes, n_holes - 1);
        goto finish;
      }
      hole_copy = GEOSGeom_clone_r(ctx, hole);
      if (hole_copy == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        destroy_geom_arr(ctx, holes, n_holes - 1);
        goto finish;
      }
      holes[n_holes] = hole_copy;
      n_holes++;
    }
    shell_copy = GEOSGeom_clone_r(ctx, shell);
    if (shell_copy == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      destroy_geom_arr(ctx, holes, n_holes - 1);
      break;
    }
    geom_arr[i] = GEOSGeom_createPolygon_r(ctx, shell_copy, holes, n_holes);
    if (geom_arr[i] == NULL) {
      // We will have a memory leak now (https://trac.osgeo.org/geos/ticket/1111)
      // but we have covered all known cases that GEOS would error by pre-emptively
      // checking if all inputs are linearrings.
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    };
  }

finish:
  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  free(geom_arr);
  if (holes != NULL) {
    free(holes);
  }
}
static PyUFuncGenericFunction polygons_funcs[1] = {&polygons_func};

static char create_collection_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void create_collection_func(char** args, npy_intp* dimensions, npy_intp* steps,
                                   void* data) {
  GEOSGeometry *g, *g_copy;
  int n_geoms, type, actual_type, expected_type, alt_expected_type;
  GEOSGeometry **temp_geoms, **geom_arr;

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  CHECK_ALLOC(geom_arr)

  // allocate a temporary array to store geometries to put in a collection
  temp_geoms = malloc(sizeof(void*) * dimensions[1]);
  CHECK_ALLOC(temp_geoms)

  GEOS_INIT_THREADS;

  BINARY_SINGLE_COREDIM_LOOP_OUTER {
    type = *(int*)ip2;
    switch (type) {
      case GEOS_MULTIPOINT:
        expected_type = GEOS_POINT;
        alt_expected_type = -1;
        break;
      case GEOS_MULTILINESTRING:
        expected_type = GEOS_LINESTRING;
        alt_expected_type = GEOS_LINEARRING;
        break;
      case GEOS_MULTIPOLYGON:
        expected_type = GEOS_POLYGON;
        alt_expected_type = -1;
        break;
      case GEOS_GEOMETRYCOLLECTION:
        expected_type = -1;
        alt_expected_type = -1;
        break;
      default:
        errstate = PGERR_GEOMETRY_TYPE;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        goto finish;
    }
    n_geoms = 0;
    cp1 = ip1;
    BINARY_SINGLE_COREDIM_LOOP_INNER {
      if (!get_geom(*(GeometryObject**)cp1, &g)) {
        errstate = PGERR_NOT_A_GEOMETRY;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        destroy_geom_arr(ctx, temp_geoms, n_geoms - 1);
        goto finish;
      }
      if (g == NULL) {
        continue;
      }
      if (expected_type != -1) {
        actual_type = GEOSGeomTypeId_r(ctx, g);
        if (actual_type == -1) {
          errstate = PGERR_GEOS_EXCEPTION;
          destroy_geom_arr(ctx, geom_arr, i - 1);
          destroy_geom_arr(ctx, temp_geoms, n_geoms - 1);
          goto finish;
        }
        if ((actual_type != expected_type) && (actual_type != alt_expected_type)) {
          errstate = PGERR_GEOMETRY_TYPE;
          destroy_geom_arr(ctx, geom_arr, i - 1);
          destroy_geom_arr(ctx, temp_geoms, n_geoms - 1);
          goto finish;
        }
      }
      g_copy = GEOSGeom_clone_r(ctx, g);
      if (g_copy == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        destroy_geom_arr(ctx, geom_arr, i - 1);
        destroy_geom_arr(ctx, temp_geoms, n_geoms - 1);
        goto finish;
      }
      temp_geoms[n_geoms] = g_copy;
      n_geoms++;
    }
    geom_arr[i] = GEOSGeom_createCollection_r(ctx, type, temp_geoms, n_geoms);
    if (geom_arr[i] == NULL) {
      // We may have a memory leak now (https://trac.osgeo.org/geos/ticket/1111)
      // but we have covered all known cases that GEOS would error by pre-emptively
      // checking if all inputs are the correct geometry types.
      errstate = PGERR_GEOS_EXCEPTION;
      destroy_geom_arr(ctx, geom_arr, i - 1);
      break;
    };
  }

finish:
  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  free(geom_arr);
  if (temp_geoms != NULL) {
    free(temp_geoms);
  }
}
static PyUFuncGenericFunction create_collection_funcs[1] = {&create_collection_func};

static char bounds_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};
static void bounds_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  GEOSGeometry *envelope = NULL, *in1;
  const GEOSGeometry* ring;
  const GEOSCoordSequence* coord_seq;
  int size;
  char *ip1 = args[0], *op1 = args[1];
  double *x1, *y1, *x2, *y2;

  GEOS_INIT_THREADS;

  npy_intp is1 = steps[0], os1 = steps[1], cs1 = steps[2];
  npy_intp n = dimensions[0], i;
  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }

    /* get the 4 (pointers to) the bbox values from the "core stride 1" (cs1) */
    x1 = (double*)(op1);
    y1 = (double*)(op1 + cs1);
    x2 = (double*)(op1 + 2 * cs1);
    y2 = (double*)(op1 + 3 * cs1);

    if (in1 == NULL) { /* no geometry => bbox becomes (nan, nan, nan, nan) */
      *x1 = *y1 = *x2 = *y2 = NPY_NAN;
    } else {
      /* construct the envelope */
      envelope = GEOSEnvelope_r(ctx, in1);
      if (envelope == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      size = GEOSGetNumCoordinates_r(ctx, envelope);

      /* get the bbox depending on the number of coordinates in the envelope */
      if (size == 0) { /* Envelope is empty */
        *x1 = *y1 = *x2 = *y2 = NPY_NAN;
      } else if (size == 1) { /* Envelope is a point */
        if (!GEOSGeomGetX_r(ctx, envelope, x1)) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        if (!GEOSGeomGetY_r(ctx, envelope, y1)) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        *x2 = *x1;
        *y2 = *y1;
      } else if (size == 5) { /* Envelope is a box */
        ring = GEOSGetExteriorRing_r(ctx, envelope);
        if (ring == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        coord_seq = GEOSGeom_getCoordSeq_r(ctx, ring);
        if (coord_seq == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        if (!GEOSCoordSeq_getX_r(ctx, coord_seq, 0, x1)) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        if (!GEOSCoordSeq_getY_r(ctx, coord_seq, 0, y1)) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        if (!GEOSCoordSeq_getX_r(ctx, coord_seq, 2, x2)) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
        if (!GEOSCoordSeq_getY_r(ctx, coord_seq, 2, y2)) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else {
        errstate = PGERR_GEOMETRY_TYPE;
        goto finish;
      }
      GEOSGeom_destroy_r(ctx, envelope);
      envelope = NULL;
    }
  }

finish:
  if (envelope != NULL) {
    GEOSGeom_destroy_r(ctx, envelope);
  }
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction bounds_funcs[1] = {&bounds_func};

/* Define the object -> geom functions (O_Y) */

static char from_wkb_dtypes[3] = {NPY_OBJECT, NPY_UINT8, NPY_OBJECT};
static void from_wkb_func(char** args, npy_intp* dimensions, npy_intp* steps,
                          void* data) {
  char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
  npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
  PyObject* in1;
  npy_uint8 on_invalid = *(npy_uint8*)ip2;
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSWKBReader* reader;
  unsigned char* wkb;
  GEOSGeometry* ret_ptr;
  Py_ssize_t size;
  char is_hex;

  if ((is2 != 0)) {
    PyErr_Format(PyExc_ValueError, "from_wkb function called with non-scalar parameters");
    return;
  }

  GEOS_INIT;

  /* Create the WKB reader */
  reader = GEOSWKBReader_create_r(ctx);
  if (reader == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    /* ip1 is pointer to array element PyObject* */
    in1 = *(PyObject**)ip1;

    if (in1 == Py_None) {
      /* None in the input propagates to the output */
      ret_ptr = NULL;
    } else {
      /* Cast the PyObject (only bytes) to char* */
      if (PyBytes_Check(in1)) {
        size = PyBytes_Size(in1);
        wkb = (unsigned char*)PyBytes_AsString(in1);
        if (wkb == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else if (PyUnicode_Check(in1)) {
        wkb = (unsigned char*)PyUnicode_AsUTF8AndSize(in1, &size);
        if (wkb == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else {
        PyErr_Format(PyExc_TypeError, "Expected bytes or string, got %s",
                     Py_TYPE(in1)->tp_name);
        goto finish;
      }

      /* Check if this is a HEX WKB */
      if (size != 0) {
        is_hex = ((wkb[0] == 48) || (wkb[0] == 49));
      } else {
        is_hex = 0;
      }

      /* Read the WKB */
      if (is_hex) {
        ret_ptr = GEOSWKBReader_readHEX_r(ctx, reader, wkb, size);
      } else {
        ret_ptr = GEOSWKBReader_read_r(ctx, reader, wkb, size);
      }
      if (ret_ptr == NULL) {
        if (on_invalid == 2) {
          // raise exception
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        } else if (on_invalid == 1) {
          // raise warning, return None
          errstate = PGWARN_INVALID_WKB;
        }
        // else: return None, no warning
      }
    }
    OUTPUT_Y;
  }

finish:
  GEOSWKBReader_destroy_r(ctx, reader);
  GEOS_FINISH;
}
static PyUFuncGenericFunction from_wkb_funcs[1] = {&from_wkb_func};

static char from_wkt_dtypes[3] = {NPY_OBJECT, NPY_UINT8, NPY_OBJECT};
static void from_wkt_func(char** args, npy_intp* dimensions, npy_intp* steps,
                          void* data) {
  char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
  npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
  PyObject* in1;
  npy_uint8 on_invalid = *(npy_uint8*)ip2;
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSGeometry* ret_ptr;
  GEOSWKTReader* reader;
  const char* wkt;

  if ((is2 != 0)) {
    PyErr_Format(PyExc_ValueError, "from_wkt function called with non-scalar parameters");
    return;
  }

  GEOS_INIT;

  /* Create the WKT reader */
  reader = GEOSWKTReader_create_r(ctx);
  if (reader == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    /* ip1 is pointer to array element PyObject* */
    in1 = *(PyObject**)ip1;

    if (in1 == Py_None) {
      /* None in the input propagates to the output */
      ret_ptr = NULL;
    } else {
      /* Cast the PyObject (bytes or str) to char* */
      if (PyBytes_Check(in1)) {
        wkt = PyBytes_AsString(in1);
        if (wkt == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else if (PyUnicode_Check(in1)) {
        wkt = PyUnicode_AsUTF8(in1);
        if (wkt == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else {
        PyErr_Format(PyExc_TypeError, "Expected bytes or string, got %s",
                     Py_TYPE(in1)->tp_name);
        goto finish;
      }

      /* Read the WKT */
      ret_ptr = GEOSWKTReader_read_r(ctx, reader, wkt);
      if (ret_ptr == NULL) {
        if (on_invalid == 2) {
          // raise exception
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        } else if (on_invalid == 1) {
          // raise warning, return None
          errstate = PGWARN_INVALID_WKT;
        }
        // else: return None, no warning
      }
    }
    OUTPUT_Y;
  }

finish:
  GEOSWKTReader_destroy_r(ctx, reader);
  GEOS_FINISH;
}
static PyUFuncGenericFunction from_wkt_funcs[1] = {&from_wkt_func};

static char from_shapely_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void from_shapely_func(char** args, npy_intp* dimensions, npy_intp* steps,
                              void* data) {
  GEOSGeometry *in_ptr, *ret_ptr;
  PyObject *in1, *attr;

  GEOS_INIT;

  UNARY_LOOP {
    /* ip1 is pointer to array element PyObject* */
    in1 = *(PyObject**)ip1;

    if (in1 == Py_None) {
      /* None in the input propagates to the output */
      ret_ptr = NULL;
    } else {
      /* Check if we have a prepared geometry */
      if (PyObject_HasAttrString(in1, "context")) {
        in1 = PyObject_GetAttrString(in1, "context");
      }
      /* Get the _geom attribute */
      attr = PyObject_GetAttrString(in1, "_geom");
      if (attr == NULL) {
        /* Raise if _geom does not exist */
        PyErr_Format(PyExc_TypeError, "Expected a shapely object or None, got %s",
                     Py_TYPE(in1)->tp_name);
        GEOS_FINISH;
        return;
      } else if (!PyLong_Check(attr)) {
        /* Raise if _geom is of incorrect type */
        PyErr_Format(PyExc_TypeError, "Expected int for the _geom attribute, got %s",
                     Py_TYPE(attr)->tp_name);
        Py_XDECREF(attr);
        GEOS_FINISH;
        return;
      }
      /* Convert it to a GEOSGeometry pointer */
      in_ptr = PyLong_AsVoidPtr(attr);
      Py_XDECREF(attr);
      /* Clone the geometry and finish */
      ret_ptr = GEOSGeom_clone_r(ctx, in_ptr);
      if (ret_ptr == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    OUTPUT_Y;
  }

finish:
  GEOS_FINISH;
}
static PyUFuncGenericFunction from_shapely_funcs[1] = {&from_shapely_func};

static char to_wkb_dtypes[6] = {NPY_OBJECT, NPY_BOOL, NPY_INT,
                                NPY_INT,    NPY_BOOL, NPY_OBJECT};
static void to_wkb_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4],
       *op1 = args[5];
  npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4],
           os1 = steps[5];
  npy_intp n = dimensions[0];
  npy_intp i;

  GEOSGeometry *in1, *temp_geom;
  GEOSWKBWriter* writer;
  unsigned char* wkb;
  size_t size;
#if !GEOS_SINCE_3_9_0
  char has_empty;
#endif  // !GEOS_SINCE_3_9_0

  if ((is2 != 0) || (is3 != 0) || (is4 != 0) || (is5 != 0)) {
    PyErr_Format(PyExc_ValueError, "to_wkb function called with non-scalar parameters");
    return;
  }

  GEOS_INIT;

  /* Create the WKB writer */
  writer = GEOSWKBWriter_create_r(ctx);
  if (writer == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  char hex = *(npy_bool*)ip2;
  GEOSWKBWriter_setOutputDimension_r(ctx, writer, *(int*)ip3);
  int byte_order = *(int*)ip4;
  if (byte_order != -1) {
    GEOSWKBWriter_setByteOrder_r(ctx, writer, *(int*)ip4);
  }
  GEOSWKBWriter_setIncludeSRID_r(ctx, writer, *(npy_bool*)ip5);

  // Check if the above functions caused a GEOS exception
  if (last_error[0] != 0) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    PyObject** out = (PyObject**)op1;

    if (in1 == NULL) {
      Py_XDECREF(*out);
      Py_INCREF(Py_None);
      *out = Py_None;
    } else {
#if !GEOS_SINCE_3_9_0
      // WKB Does not allow empty points in GEOS<3.9.
      // We check for that and patch the POINT EMPTY if necessary
      has_empty = has_point_empty(ctx, in1);
      if (has_empty == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      if (has_empty) {
        temp_geom = point_empty_to_nan_all_geoms(ctx, in1);
      } else {
        temp_geom = in1;
      }
#else
      temp_geom = in1;
#endif  // !GEOS_SINCE_3_9_0
      if (hex) {
        wkb = GEOSWKBWriter_writeHEX_r(ctx, writer, temp_geom, &size);
      } else {
        wkb = GEOSWKBWriter_write_r(ctx, writer, temp_geom, &size);
      }
#if !GEOS_SINCE_3_9_0
      // Destroy the temp_geom if it was patched (POINT EMPTY patch)
      if (has_empty) {
        GEOSGeom_destroy_r(ctx, temp_geom);
      }
#endif  // !GEOS_SINCE_3_9_0
      if (wkb == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      Py_XDECREF(*out);
      if (hex) {
        *out = PyUnicode_FromStringAndSize((char*)wkb, size);
      } else {
        *out = PyBytes_FromStringAndSize((char*)wkb, size);
      }
      GEOSFree_r(ctx, wkb);
    }
  }

finish:
  GEOSWKBWriter_destroy_r(ctx, writer);
  GEOS_FINISH;
}
static PyUFuncGenericFunction to_wkb_funcs[1] = {&to_wkb_func};

static char to_wkt_dtypes[6] = {NPY_OBJECT, NPY_INT,  NPY_BOOL,
                                NPY_INT,    NPY_BOOL, NPY_OBJECT};
static void to_wkt_func(char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
  char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4],
       *op1 = args[5];
  npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4],
           os1 = steps[5];
  npy_intp n = dimensions[0];
  npy_intp i;

  GEOSGeometry* in1;
  GEOSWKTWriter* writer;
  char* wkt;

  if ((is2 != 0) || (is3 != 0) || (is4 != 0) || (is5 != 0)) {
    PyErr_Format(PyExc_ValueError, "to_wkt function called with non-scalar parameters");
    return;
  }

  GEOS_INIT;

  /* Create the WKT writer */
  writer = GEOSWKTWriter_create_r(ctx);
  if (writer == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }
  GEOSWKTWriter_setRoundingPrecision_r(ctx, writer, *(int*)ip2);
  GEOSWKTWriter_setTrim_r(ctx, writer, *(npy_bool*)ip3);
  GEOSWKTWriter_setOutputDimension_r(ctx, writer, *(int*)ip4);
  GEOSWKTWriter_setOld3D_r(ctx, writer, *(npy_bool*)ip5);

  // Check if the above functions caused a GEOS exception
  if (last_error[0] != 0) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    PyObject** out = (PyObject**)op1;

    if (in1 == NULL) {
      Py_XDECREF(*out);
      Py_INCREF(Py_None);
      *out = Py_None;
    } else {
#if GEOS_SINCE_3_9_0
      errstate = wkt_empty_3d_geometry(ctx, in1, &wkt);
      if (errstate != PGERR_SUCCESS) {
        goto finish;
      }
      if (wkt != NULL) {
        *out = PyUnicode_FromString(wkt);
        goto finish;
      }

#else
      // Before GEOS 3.9.0, there was as segfault on e.g. MULTIPOINT (1 1, EMPTY)
      errstate = check_to_wkt_compatible(ctx, in1);
      if (errstate != PGERR_SUCCESS) {
        goto finish;
      }
#endif
      wkt = GEOSWKTWriter_write_r(ctx, writer, in1);
      if (wkt == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      Py_XDECREF(*out);
      *out = PyUnicode_FromString(wkt);
      GEOSFree_r(ctx, wkt);
    }
  }

finish:
  GEOSWKTWriter_destroy_r(ctx, writer);
  GEOS_FINISH;
}
static PyUFuncGenericFunction to_wkt_funcs[1] = {&to_wkt_func};

#if GEOS_SINCE_3_10_0

static char from_geojson_dtypes[3] = {NPY_OBJECT, NPY_UINT8, NPY_OBJECT};
static void from_geojson_func(char** args, npy_intp* dimensions, npy_intp* steps,
                              void* data) {
  char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
  npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
  PyObject* in1;
  npy_uint8 on_invalid = *(npy_uint8*)ip2;
  npy_intp n = dimensions[0];
  npy_intp i;
  GEOSGeometry* ret_ptr;
  GEOSGeoJSONReader* reader;
  const char* geojson;

  if ((is2 != 0)) {
    PyErr_Format(PyExc_ValueError,
                 "from_geojson function called with non-scalar parameters");
    return;
  }

  GEOS_INIT;

  /* Create the WKT reader */
  reader = GEOSGeoJSONReader_create_r(ctx);
  if (reader == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    /* ip1 is pointer to array element PyObject* */
    in1 = *(PyObject**)ip1;

    if (in1 == Py_None) {
      /* None in the input propagates to the output */
      ret_ptr = NULL;
    } else {
      /* Cast the PyObject (bytes or str) to char* */
      if (PyBytes_Check(in1)) {
        geojson = PyBytes_AsString(in1);
        if (geojson == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else if (PyUnicode_Check(in1)) {
        geojson = PyUnicode_AsUTF8(in1);
        if (geojson == NULL) {
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        }
      } else {
        PyErr_Format(PyExc_TypeError, "Expected bytes or string, got %s",
                     Py_TYPE(in1)->tp_name);
        goto finish;
      }

      /* Read the GeoJSON */
      ret_ptr = GEOSGeoJSONReader_readGeometry_r(ctx, reader, geojson);
      if (ret_ptr == NULL) {
        if (on_invalid == 2) {
          // raise exception
          errstate = PGERR_GEOS_EXCEPTION;
          goto finish;
        } else if (on_invalid == 1) {
          // raise warning, return None
          errstate = PGWARN_INVALID_GEOJSON;
        }
        // else: return None, no warning
      }
    }
    OUTPUT_Y;
  }

finish:
  GEOSGeoJSONReader_destroy_r(ctx, reader);
  GEOS_FINISH;
}
static PyUFuncGenericFunction from_geojson_funcs[1] = {&from_geojson_func};

static char to_geojson_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void to_geojson_func(char** args, npy_intp* dimensions, npy_intp* steps,
                            void* data) {
  char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
  npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
  npy_intp n = dimensions[0];
  npy_intp i;

  GEOSGeometry* in1;
  int indent;
  GEOSGeoJSONWriter* writer;
  char* geojson;
  char point_empty_error;

  if (is2 != 0) {
    PyErr_Format(PyExc_ValueError, "to_geojson indent parameter must be a scalar");
    return;
  }
  indent = *(int*)ip2;

  GEOS_INIT;

  /* Create the GeoJSON writer */
  writer = GEOSGeoJSONWriter_create_r(ctx);
  if (writer == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    goto finish;
  }

  for (i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    PyObject** out = (PyObject**)op1;

    if (in1 == NULL) {
      Py_XDECREF(*out);
      Py_INCREF(Py_None);
      *out = Py_None;
    } else {
      // Check for empty points (https://trac.osgeo.org/geos/ticket/1139)
      point_empty_error = has_point_empty(ctx, in1);
      if (point_empty_error == 2) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      } else if (point_empty_error == 1) {
        errstate = PGERR_GEOJSON_EMPTY_POINT;
        goto finish;
      }
      geojson = GEOSGeoJSONWriter_writeGeometry_r(ctx, writer, in1, indent);
      if (geojson == NULL) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
      Py_XDECREF(*out);
      *out = PyUnicode_FromString(geojson);
      GEOSFree_r(ctx, geojson);
    }
  }

finish:
  GEOSGeoJSONWriter_destroy_r(ctx, writer);
  GEOS_FINISH;
}
static PyUFuncGenericFunction to_geojson_funcs[1] = {&to_geojson_func};

#endif  // GEOS_SINCE_3_10_0

/*
TODO polygonizer functions
TODO prepared geometry predicate functions
TODO relate functions
*/

#define DEFINE_Y_b(NAME)                                                       \
  ufunc = PyUFunc_FromFuncAndData(Y_b_funcs, NAME##_data, Y_b_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, NULL, 0);               \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_O_b(NAME)                                                       \
  ufunc = PyUFunc_FromFuncAndData(O_b_funcs, NAME##_data, O_b_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, NULL, 0);               \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YY_b(NAME)                                                        \
  ufunc = PyUFunc_FromFuncAndData(YY_b_funcs, NAME##_data, YY_b_dtypes, 1, 2, 1, \
                                  PyUFunc_None, #NAME, "", 0);                   \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YY_b_p(NAME)                                                          \
  ufunc = PyUFunc_FromFuncAndData(YY_b_p_funcs, NAME##_data, YY_b_p_dtypes, 1, 2, 1, \
                                  PyUFunc_None, #NAME, "", 0);                       \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Y_Y(NAME)                                                       \
  ufunc = PyUFunc_FromFuncAndData(Y_Y_funcs, NAME##_data, Y_Y_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, "", 0);                 \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Y(NAME)                                                                   \
  ufunc = PyUFunc_FromFuncAndData(Y_funcs, NAME##_data, Y_dtypes, 1, 1, 0, PyUFunc_None, \
                                  #NAME, "", 0);                                         \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Yi_Y(NAME)                                                        \
  ufunc = PyUFunc_FromFuncAndData(Yi_Y_funcs, NAME##_data, Yi_Y_dtypes, 1, 2, 1, \
                                  PyUFunc_None, #NAME, "", 0);                   \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Yd_Y(NAME)                                                        \
  ufunc = PyUFunc_FromFuncAndData(Yd_Y_funcs, NAME##_data, Yd_Y_dtypes, 1, 2, 1, \
                                  PyUFunc_None, #NAME, "", 0);                   \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YY_Y(NAME)                                                        \
  ufunc = PyUFunc_FromFuncAndData(YY_Y_funcs, NAME##_data, YY_Y_dtypes, 1, 2, 1, \
                                  PyUFunc_None, #NAME, "", 0);                   \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YY_Y_REORDERABLE(NAME)                                            \
  ufunc = PyUFunc_FromFuncAndData(YY_Y_funcs, NAME##_data, YY_Y_dtypes, 1, 2, 1, \
                                  PyUFunc_ReorderableNone, #NAME, "", 0);        \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Y_d(NAME)                                                       \
  ufunc = PyUFunc_FromFuncAndData(Y_d_funcs, NAME##_data, Y_d_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, "", 0);                 \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Y_B(NAME)                                                       \
  ufunc = PyUFunc_FromFuncAndData(Y_B_funcs, NAME##_data, Y_B_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, "", 0);                 \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_Y_i(NAME)                                                       \
  ufunc = PyUFunc_FromFuncAndData(Y_i_funcs, NAME##_data, Y_i_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, "", 0);                 \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YY_d(NAME)                                                        \
  ufunc = PyUFunc_FromFuncAndData(YY_d_funcs, NAME##_data, YY_d_dtypes, 1, 2, 1, \
                                  PyUFunc_None, #NAME, "", 0);                   \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YYd_d(NAME)                                                         \
  ufunc = PyUFunc_FromFuncAndData(YYd_d_funcs, NAME##_data, YYd_d_dtypes, 1, 3, 1, \
                                  PyUFunc_None, #NAME, "", 0);                     \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_YYd_Y(NAME)                                                         \
  ufunc = PyUFunc_FromFuncAndData(YYd_Y_funcs, NAME##_data, YYd_Y_dtypes, 1, 3, 1, \
                                  PyUFunc_None, #NAME, "", 0);                     \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_CUSTOM(NAME, N_IN)                                                     \
  ufunc = PyUFunc_FromFuncAndData(NAME##_funcs, null_data, NAME##_dtypes, 1, N_IN, 1, \
                                  PyUFunc_None, #NAME, "", 0);                        \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_GENERALIZED(NAME, N_IN, SIGNATURE)                                        \
  ufunc = PyUFunc_FromFuncAndDataAndSignature(NAME##_funcs, null_data, NAME##_dtypes, 1, \
                                              N_IN, 1, PyUFunc_None, #NAME, "", 0,       \
                                              SIGNATURE);                                \
  PyDict_SetItemString(d, #NAME, ufunc)

#define DEFINE_GENERALIZED_NOUT4(NAME, N_IN, SIGNATURE)                                  \
  ufunc = PyUFunc_FromFuncAndDataAndSignature(NAME##_funcs, null_data, NAME##_dtypes, 1, \
                                              N_IN, 4, PyUFunc_None, #NAME, "", 0,       \
                                              SIGNATURE);                                \
  PyDict_SetItemString(d, #NAME, ufunc)

int init_ufuncs(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  DEFINE_Y_b(is_empty);
  DEFINE_Y_b(is_simple);
  DEFINE_Y_b(is_geometry);
  DEFINE_Y_b(is_ring);
  DEFINE_Y_b(has_z);
  DEFINE_Y_b(is_closed);
  DEFINE_Y_b(is_valid);

  DEFINE_O_b(is_geometry);
  DEFINE_O_b(is_missing);
  DEFINE_O_b(is_valid_input);

  DEFINE_YY_b_p(disjoint);
  DEFINE_YY_b_p(touches);
  DEFINE_YY_b_p(intersects);
  DEFINE_YY_b_p(crosses);
  DEFINE_YY_b_p(within);
  DEFINE_YY_b_p(contains);
  DEFINE_YY_b_p(contains_properly);
  DEFINE_YY_b_p(overlaps);
  DEFINE_YY_b(equals);
  DEFINE_YY_b_p(covers);
  DEFINE_YY_b_p(covered_by);
  DEFINE_CUSTOM(is_prepared, 1);

  DEFINE_Y_Y(envelope);
  DEFINE_Y_Y(convex_hull);
  DEFINE_Y_Y(boundary);
  DEFINE_Y_Y(unary_union);
  DEFINE_Y_Y(point_on_surface);
  DEFINE_Y_Y(centroid);
  DEFINE_Y_Y(line_merge);
  DEFINE_Y_Y(extract_unique_points);
  DEFINE_Y_Y(get_exterior_ring);
  DEFINE_Y_Y(normalize);
  DEFINE_Y_Y(force_2d);

  DEFINE_Y(prepare);
  DEFINE_Y(destroy_prepared);

  DEFINE_Yi_Y(get_point);
  DEFINE_Yi_Y(get_interior_ring);
  DEFINE_Yi_Y(get_geometry);
  DEFINE_Yi_Y(set_srid);

  DEFINE_Yd_Y(line_interpolate_point);
  DEFINE_Yd_Y(line_interpolate_point_normalized);
  DEFINE_Yd_Y(simplify);
  DEFINE_Yd_Y(simplify_preserve_topology);
  DEFINE_Yd_Y(force_3d);

  // 'REORDERABLE' sets PyUFunc_ReorderableNone instead of PyUFunc_None as 'identity',
  // meaning that the order of arguments does not matter. This enables reduction over
  // multiple (or all) axes.
  DEFINE_YY_Y_REORDERABLE(intersection);
  DEFINE_YY_Y(difference);
  DEFINE_YY_Y_REORDERABLE(symmetric_difference);
  DEFINE_YY_Y_REORDERABLE(union);
  DEFINE_YY_Y(shared_paths);

  DEFINE_Y_d(get_x);
  DEFINE_Y_d(get_y);
  DEFINE_Y_d(area);
  DEFINE_Y_d(length);

  DEFINE_Y_i(get_type_id);
  DEFINE_Y_i(get_dimensions);
  DEFINE_Y_i(get_coordinate_dimension);
  DEFINE_Y_i(get_srid);
  DEFINE_Y_i(get_num_points);
  DEFINE_Y_i(get_num_interior_rings);
  DEFINE_Y_i(get_num_geometries);
  DEFINE_Y_i(get_num_coordinates);

  DEFINE_YY_d(distance);
  DEFINE_YY_d(hausdorff_distance);
  DEFINE_YY_d(line_locate_point);
  DEFINE_YY_d(line_locate_point_normalized);

  DEFINE_YYd_d(hausdorff_distance_densify);

  DEFINE_CUSTOM(box, 5);
  DEFINE_CUSTOM(buffer, 7);
  DEFINE_CUSTOM(offset_curve, 5);
  DEFINE_CUSTOM(snap, 3);
  DEFINE_CUSTOM(clip_by_rect, 5);
  DEFINE_CUSTOM(equals_exact, 3);

  DEFINE_CUSTOM(delaunay_triangles, 3);
  DEFINE_CUSTOM(voronoi_polygons, 4);
  DEFINE_CUSTOM(is_valid_reason, 1);
  DEFINE_CUSTOM(relate, 2);
  DEFINE_CUSTOM(relate_pattern, 3);
  DEFINE_GENERALIZED(polygonize, 1, "(d)->()");
  DEFINE_GENERALIZED_NOUT4(polygonize_full, 1, "(d)->(),(),(),()");
  DEFINE_CUSTOM(shortest_line, 2);

  DEFINE_GENERALIZED(points, 1, "(d)->()");
  DEFINE_GENERALIZED(linestrings, 1, "(i, d)->()");
  DEFINE_GENERALIZED(linearrings, 1, "(i, d)->()");
  DEFINE_GENERALIZED(bounds, 1, "()->(n)");
  DEFINE_GENERALIZED(polygons, 2, "(),(i)->()");
  DEFINE_GENERALIZED(create_collection, 2, "(i),()->()");

  DEFINE_CUSTOM(from_wkb, 2);
  DEFINE_CUSTOM(from_wkt, 2);
  DEFINE_CUSTOM(to_wkb, 5);
  DEFINE_CUSTOM(to_wkt, 5);
  DEFINE_CUSTOM(from_shapely, 1);

#if GEOS_SINCE_3_6_0
  DEFINE_Y_d(minimum_clearance);
  DEFINE_Y_d(get_precision);
  DEFINE_Y_Y(oriented_envelope);
  DEFINE_CUSTOM(set_precision, 3);
#endif

#if GEOS_SINCE_3_7_0
  DEFINE_Y_b(is_ccw);
  DEFINE_Y_d(get_z);
  DEFINE_Y_Y(reverse);
  DEFINE_YY_d(frechet_distance);
  DEFINE_YYd_d(frechet_distance_densify);
#endif

#if GEOS_SINCE_3_8_0
  DEFINE_Y_Y(make_valid);
  DEFINE_Y_Y(build_area);
  DEFINE_Y_Y(coverage_union);
  DEFINE_Y_Y(minimum_bounding_circle);
  DEFINE_Y_d(minimum_bounding_radius);
#endif

#if GEOS_SINCE_3_9_0
  DEFINE_YYd_Y(difference_prec);
  DEFINE_YYd_Y(intersection_prec);
  DEFINE_YYd_Y(symmetric_difference_prec);
  DEFINE_YYd_Y(union_prec);
  DEFINE_Yd_Y(unary_union_prec);
#endif

#if GEOS_SINCE_3_10_0
  DEFINE_Yd_Y(segmentize);
  DEFINE_CUSTOM(dwithin, 3);
  DEFINE_CUSTOM(from_geojson, 2);
  DEFINE_CUSTOM(to_geojson, 2);
#endif

  Py_DECREF(ufunc);
  return 0;
}
