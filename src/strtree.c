#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <float.h>
#include <structmember.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "geos.h"
#include "kvec.h"
#include "pygeom.h"
#include "strtree.h"
#include "vector.h"

/* GEOS function that takes a prepared geometry and a regular geometry
 * and returns bool value */

typedef char FuncGEOS_YpY_b(void* context, const GEOSPreparedGeometry* a,
                            const GEOSGeometry* b);

/* get predicate function based on ID.  See strtree.py::BinaryPredicate for
 * lookup table of id to function name */

FuncGEOS_YpY_b* get_predicate_func(int predicate_id) {
  switch (predicate_id) {
    case 1: {  // intersects
      return (FuncGEOS_YpY_b*)GEOSPreparedIntersects_r;
    }
    case 2: {  // within
      return (FuncGEOS_YpY_b*)GEOSPreparedWithin_r;
    }
    case 3: {  // contains
      return (FuncGEOS_YpY_b*)GEOSPreparedContains_r;
    }
    case 4: {  // overlaps
      return (FuncGEOS_YpY_b*)GEOSPreparedOverlaps_r;
    }
    case 5: {  // crosses
      return (FuncGEOS_YpY_b*)GEOSPreparedCrosses_r;
    }
    case 6: {  // touches
      return (FuncGEOS_YpY_b*)GEOSPreparedTouches_r;
    }
    case 7: {  // covers
      return (FuncGEOS_YpY_b*)GEOSPreparedCovers_r;
    }
    case 8: {  // covered_by
      return (FuncGEOS_YpY_b*)GEOSPreparedCoveredBy_r;
    }
    case 9: {  // contains_properly
      return (FuncGEOS_YpY_b*)GEOSPreparedContainsProperly_r;
    }
    default: {  // unknown predicate
      PyErr_SetString(PyExc_ValueError, "Invalid query predicate");
      return NULL;
    }
  }
}

/* Calculate indices of tree geometries.
 * This uses pointer offsets of geometries from the head of tree geometries to calculate
 * corresponding indices.
 *
 * Parameters
 * ----------
 * tree_geoms: array of tree geometries
 *
 * arr: dynamic vector of addresses of geometries within tree geometries array
 */
static PyArrayObject* tree_geom_offsets_to_npy_arr(GeometryObject** tree_geoms,
                                                   tree_geom_vec_t* geoms) {
  size_t i;
  size_t size = kv_size(*geoms);
  npy_intp geom_index;
  char* head_ptr = (char*)tree_geoms;  // head of tree geometry array

  npy_intp dims[1] = {size};
  // the following raises a compiler warning based on how the macro is defined
  // in numpy.  There doesn't appear to be anything we can do to avoid it.
  PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INTP);
  if (result == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
    return NULL;
  }

  for (i = 0; i < size; i++) {
    // Calculate index using offset of its address compared to head of tree geometries
    geom_index =
        (npy_intp)(((char*)kv_A(*geoms, i) - head_ptr) / sizeof(GeometryObject*));

    // assign value into numpy array
    *(npy_intp*)PyArray_GETPTR1(result, i) = geom_index;
  }

  return (PyArrayObject*)result;
}

static void STRtree_dealloc(STRtreeObject* self) {
  size_t i;

  // free the tree
  if (self->ptr != NULL) {
    GEOS_INIT;
    GEOSSTRtree_destroy_r(ctx, self->ptr);
    GEOS_FINISH;
  }
  // free the geometries
  for (i = 0; i < self->_geoms_size; i++) {
    Py_XDECREF(self->_geoms[i]);
  }

  free(self->_geoms);
  // free the PyObject
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* STRtree_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  int node_capacity;
  PyObject* arr;
  void *tree, *ptr;
  npy_intp n, i, counter = 0, count_indexed = 0;
  GEOSGeometry* geom;
  GeometryObject* obj;
  GeometryObject** _geoms;

  if (!PyArg_ParseTuple(args, "Oi", &arr, &node_capacity)) {
    return NULL;
  }
  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Not an ndarray");
    return NULL;
  }
  if (!PyArray_ISOBJECT((PyArrayObject*)arr)) {
    PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
    return NULL;
  }
  if (PyArray_NDIM((PyArrayObject*)arr) != 1) {
    PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
    return NULL;
  }

  GEOS_INIT;

  tree = GEOSSTRtree_create_r(ctx, (size_t)node_capacity);
  if (tree == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    return NULL;
    GEOS_FINISH;
  }

  n = PyArray_SIZE((PyArrayObject*)arr);

  _geoms = (GeometryObject**)malloc(n * sizeof(GeometryObject*));

  for (i = 0; i < n; i++) {
    /* get the geometry */
    ptr = PyArray_GETPTR1((PyArrayObject*)arr, i);
    obj = *(GeometryObject**)ptr;
    /* fail and cleanup incase obj was no geometry */
    if (!get_geom(obj, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      GEOSSTRtree_destroy_r(ctx, tree);

      // free the geometries
      for (i = 0; i < counter; i++) {
        Py_XDECREF(_geoms[i]);
      }
      free(_geoms);
      GEOS_FINISH;
      return NULL;
    }
    /* If geometry is None or empty, do not add it to the tree or count.
     * Set it as NULL for the internal geometries used for predicate tests.
     */
    if (geom == NULL || GEOSisEmpty_r(ctx, geom)) {
      _geoms[i] = NULL;
    } else {
      // NOTE: we must keep a reference to the GeometryObject added to the tree in order
      // to avoid segfaults later.  See: https://github.com/pygeos/pygeos/pull/100.
      Py_INCREF(obj);
      _geoms[i] = obj;
      count_indexed++;

      // Store the address of this geometry within _geoms array as the item data in the
      // tree.  This address is used to calculate the original index of the geometry in
      // the input array.
      // NOTE: the type of item data we store is GeometryObject**.
      GEOSSTRtree_insert_r(ctx, tree, geom, &(_geoms[i]));
    }
    counter++;
  }

  STRtreeObject* self = (STRtreeObject*)type->tp_alloc(type, 0);
  if (self == NULL) {
    GEOSSTRtree_destroy_r(ctx, tree);
    GEOS_FINISH;
    return NULL;
  }
  GEOS_FINISH;
  self->ptr = tree;
  self->count = count_indexed;
  self->_geoms_size = n;
  self->_geoms = _geoms;
  return (PyObject*)self;
}

/* Callback called by strtree_query with item data of each intersecting geometry
 * and a dynamic vector to push that item onto.
 *
 * Item data is the address of that geometry within the tree geometries (_geoms) array.
 *
 * Parameters
 * ----------
 * item: index of intersected geometry in the tree
 *
 * user_data: pointer to dynamic vector
 * */
void query_callback(void* item, void* user_data) {
  kv_push(GeometryObject**, *(tree_geom_vec_t*)user_data, item);
}

/* Evaluate the predicate function against a prepared version of geom
 * for each geometry in the tree specified by indexes in out_indexes.
 * out_indexes is updated in place with the indexes of the geometries in the
 * tree that meet the predicate.
 *
 * Parameters
 * ----------
 * predicate_func: pointer to a prepared predicate function, e.g.,
 *   GEOSPreparedIntersects_r
 *
 * geom: input geometry to prepare and test against each geometry in the tree specified by
 *   in_indexes.
 *
 * prepared_geom: input prepared geometry, only if previously created.  If NULL, geom
 *   will be prepared instead.
 *
 * in_geoms: pointer to dynamic vector of addresses in tree geometries (_geoms) that have
 *   overlapping envelopes with envelope of input geometry.
 *
 * out_geoms: pointer to dynamic vector of addresses in tree geometries (_geoms) that meet
 *   predicate function.
 *
 * count: pointer to an integer where the number of geometries that met the predicate will
 *   be written.
 *
 * Returns PGERR_GEOS_EXCEPTION if an error was encountered or PGERR_SUCCESS otherwise
 * */

static char evaluate_predicate(void* context, FuncGEOS_YpY_b* predicate_func,
                               GEOSGeometry* geom, GEOSPreparedGeometry* prepared_geom,
                               tree_geom_vec_t* in_geoms, tree_geom_vec_t* out_geoms,
                               npy_intp* count) {
  GeometryObject* pg_geom;
  GeometryObject** pg_geom_loc;  // address of geometry in tree geometries (_geoms)
  GEOSGeometry* target_geom;
  const GEOSPreparedGeometry* prepared_geom_tmp;
  npy_intp i, size;

  if (prepared_geom == NULL) {
    // geom was not previously prepared, prepare it now
    prepared_geom_tmp = GEOSPrepare_r(context, geom);
    if (prepared_geom_tmp == NULL) {
      return PGERR_GEOS_EXCEPTION;
    }
  } else {
    // cast to const only needed until larger refactor of all geom pointers to const
    prepared_geom_tmp = (const GEOSPreparedGeometry*)prepared_geom;
  }

  size = kv_size(*in_geoms);
  *count = 0;
  for (i = 0; i < size; i++) {
    // get address of geometry in tree geometries, then use that to get associated
    // GEOS geometry
    pg_geom_loc = kv_A(*in_geoms, i);
    pg_geom = *pg_geom_loc;

    if (pg_geom == NULL) {
      continue;
    }
    get_geom(pg_geom, &target_geom);

    // keep the geometry if it passes the predicate
    if (predicate_func(context, prepared_geom_tmp, target_geom)) {
      kv_push(GeometryObject**, *out_geoms, pg_geom_loc);
      (*count)++;
    }
  }

  if (prepared_geom == NULL) {
    // only if we created prepared_geom_tmp here, destroy it
    GEOSPreparedGeom_destroy_r(context, prepared_geom_tmp);
    prepared_geom_tmp = NULL;
  }

  return PGERR_SUCCESS;
}

/* Query the tree based on input geometry and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned.
 *
 * args must be:
 * - pygeos geometry object
 * - predicate id (see strtree.py for list of ids)
 * */

static PyObject* STRtree_query(STRtreeObject* self, PyObject* args) {
  GeometryObject* geometry = NULL;
  int predicate_id = 0;  // default no predicate
  GEOSGeometry* geom = NULL;
  GEOSPreparedGeometry* prepared_geom = NULL;
  npy_intp count;
  FuncGEOS_YpY_b* predicate_func = NULL;
  PyArrayObject* result;

  // Addresses in tree geometries (_geoms) that match tree
  tree_geom_vec_t query_geoms;

  // Addresses in tree geometries (_geoms) that meet predicate (if present)
  tree_geom_vec_t predicate_geoms;

  if (self->ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
    return NULL;
  }

  if (!PyArg_ParseTuple(args, "O!i", &GeometryType, &geometry, &predicate_id)) {
    return NULL;
  }

  if (!get_geom_with_prepared(geometry, &geom, &prepared_geom)) {
    PyErr_SetString(PyExc_TypeError, "Invalid geometry");
    return NULL;
  }

  if (self->count == 0) {
    npy_intp dims[1] = {0};
    return PyArray_SimpleNew(1, dims, NPY_INTP);
  }

  if (predicate_id != 0) {
    predicate_func = get_predicate_func(predicate_id);
    if (predicate_func == NULL) {
      return NULL;
    }
  }

  GEOS_INIT;

  // query the tree for addresses of tree geometries (_geoms) in the tree with
  // envelopes that intersect the geometry.
  kv_init(query_geoms);
  if (geom != NULL && !GEOSisEmpty_r(ctx, geom)) {
    GEOSSTRtree_query_r(ctx, self->ptr, geom, query_callback, &query_geoms);
  }

  if (predicate_id == 0 || kv_size(query_geoms) == 0) {
    // No predicate function provided, return all geometry indexes from
    // query.  If array is empty, return an empty numpy array
    result = tree_geom_offsets_to_npy_arr(self->_geoms, &query_geoms);
    kv_destroy(query_geoms);
    GEOS_FINISH;
    return (PyObject*)result;
  }

  kv_init(predicate_geoms);
  errstate = evaluate_predicate(ctx, predicate_func, geom, prepared_geom, &query_geoms,
                                &predicate_geoms, &count);
  if (errstate != PGERR_SUCCESS) {
    // error performing predicate
    kv_destroy(query_geoms);
    kv_destroy(predicate_geoms);
    GEOS_FINISH;
    return NULL;
  }

  // calculate indices of tree geometries and output to array
  result = tree_geom_offsets_to_npy_arr(self->_geoms, &predicate_geoms);

  kv_destroy(query_geoms);
  kv_destroy(predicate_geoms);

  GEOS_FINISH;
  return (PyObject*)result;
}

/* Query the tree based on input geometries and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned.
 * Returns two arrays of equal length: first is indexes of the source geometries
 * and second is indexes of tree geometries that meet the above conditions.
 *
 * args must be:
 * - ndarray of pygeos geometries
 * - predicate id (see strtree.py for list of ids)
 *
 * */

static PyObject* STRtree_query_bulk(STRtreeObject* self, PyObject* args) {
  PyObject* arr;
  PyArrayObject* pg_geoms;
  GeometryObject* pg_geom = NULL;
  int predicate_id = 0;  // default no predicate
  GEOSGeometry* geom = NULL;
  GEOSPreparedGeometry* prepared_geom = NULL;
  index_vec_t src_indexes;  // Indices of input geometries
  npy_intp i, j, n, size, geom_index;
  FuncGEOS_YpY_b* predicate_func = NULL;
  char* head_ptr = (char*)self->_geoms;
  PyArrayObject* result;

  // Addresses in tree geometries (_geoms) that match tree
  tree_geom_vec_t query_geoms;

  // Aggregated addresses in tree geometries (_geoms) that also meet predicate (if
  // present)
  tree_geom_vec_t target_geoms;

  if (self->ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
    return NULL;
  }

  if (!PyArg_ParseTuple(args, "Oi", &arr, &predicate_id)) {
    return NULL;
  }

  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Not an ndarray");
    return NULL;
  }

  pg_geoms = (PyArrayObject*)arr;
  if (!PyArray_ISOBJECT(pg_geoms)) {
    PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
    return NULL;
  }

  if (PyArray_NDIM(pg_geoms) != 1) {
    PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
    return NULL;
  }

  if (predicate_id != 0) {
    predicate_func = get_predicate_func(predicate_id);
    if (predicate_func == NULL) {
      return NULL;
    }
  }

  n = PyArray_SIZE(pg_geoms);

  if (self->count == 0 || n == 0) {
    npy_intp dims[2] = {2, 0};
    return PyArray_SimpleNew(2, dims, NPY_INTP);
  }

  kv_init(src_indexes);
  kv_init(target_geoms);

  GEOS_INIT_THREADS;

  for (i = 0; i < n; i++) {
    // get pygeos geometry from input geometry array
    pg_geom = *(GeometryObject**)PyArray_GETPTR1(pg_geoms, i);
    if (!get_geom_with_prepared(pg_geom, &geom, &prepared_geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      break;
    }
    if (geom == NULL || GEOSisEmpty_r(ctx, geom)) {
      continue;
    }

    kv_init(query_geoms);
    GEOSSTRtree_query_r(ctx, self->ptr, geom, query_callback, &query_geoms);

    if (kv_size(query_geoms) == 0) {
      // no target geoms in query window, skip this source geom
      kv_destroy(query_geoms);
      continue;
    }

    if (predicate_id == 0) {
      // no predicate, push results directly onto target_geoms
      size = kv_size(query_geoms);
      for (j = 0; j < size; j++) {
        // push index of source geometry onto src_indexes
        kv_push(npy_intp, src_indexes, i);

        // push geometry that matched tree onto target_geoms
        kv_push(GeometryObject**, target_geoms, kv_A(query_geoms, j));
      }
    } else {
      // Tree geometries that meet the predicate are pushed onto target_geoms
      errstate = evaluate_predicate(ctx, predicate_func, geom, prepared_geom,
                                    &query_geoms, &target_geoms, &size);

      if (errstate != PGERR_SUCCESS) {
        kv_destroy(query_geoms);
        kv_destroy(src_indexes);
        kv_destroy(target_geoms);
        break;
      }

      for (j = 0; j < size; j++) {
        kv_push(npy_intp, src_indexes, i);
      }
    }

    kv_destroy(query_geoms);
  }

  GEOS_FINISH_THREADS;

  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  size = kv_size(src_indexes);
  npy_intp dims[2] = {2, size};

  // the following raises a compiler warning based on how the macro is defined
  // in numpy.  There doesn't appear to be anything we can do to avoid it.
  result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INTP);
  if (result == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
    return NULL;
  }

  for (i = 0; i < size; i++) {
    // assign value into numpy arrays
    *(npy_intp*)PyArray_GETPTR2(result, 0, i) = kv_A(src_indexes, i);

    // Calculate index using offset of its address compared to head of _geoms
    geom_index =
        (npy_intp)(((char*)kv_A(target_geoms, i) - head_ptr) / sizeof(GeometryObject*));
    *(npy_intp*)PyArray_GETPTR2(result, 1, i) = geom_index;
  }

  kv_destroy(src_indexes);
  kv_destroy(target_geoms);
  return (PyObject*)result;
}

/* Calculate the distance between items in the tree and the src_geom.
 * Note: this is only called by the tree after first evaluating the overlap between
 * the the envelope of a tree node and the query geometry.  It may not be called for
 * all equidistant results.
 *
 * In order to force GEOS to check neighbors in adjacent tree nodes, a slight adjustment
 * is added to the distance returned to the tree.  Otherwise, the tree nearest neighbor
 * algorithm terminates if the envelopes of adjacent tree nodes are not further than
 * this distance, and not all equidistant or intersected neighbors are checked.  The
 * accurate distances are stored into the distance pairs in userdata.
 *
 * Parameters
 * ----------
 * item1: address of geometry in tree geometries (_geoms)
 *
 * item2: pointer to GeometryObject* of query geometry
 *
 * distance: pointer to distance that gets updated in this function
 *
 * userdata: instance of tree_nearest_userdata_t, includes vector to cache nearest
 *   distance pairs visited by this function, GEOS context handle, and minimum observed
 *   distance.
 *
 * Returns
 * -------
 * 0 on error (caller immediately exits and returns NULL); 1 on success
 * */
int distance_callback(const void* item1, const void* item2, double* distance,
                      void* userdata) {
  GEOSGeometry* tree_geom = NULL;
  GEOSGeometry* query_geom = NULL;
  size_t pairs_size;
  double calc_distance;

  GeometryObject* tree_pg_geom = *((GeometryObject**)item1);
  // Note: this is guarded for NULL during construction of tree; no need to check here.
  get_geom(tree_pg_geom, &tree_geom);

  GeometryObject* query_pg_geom = *((GeometryObject**)item2);
  // Note: this is guarded for NULL in calling function; no need to check here.
  get_geom(query_pg_geom, &query_geom);

  tree_nearest_userdata_t* params = (tree_nearest_userdata_t*)userdata;

  // distance returns 1 on success, 0 on error
  if (GEOSDistance_r(params->ctx, query_geom, tree_geom, &calc_distance) == 0) {
    return 0;
  }

  // store any pairs that are smaller than the minimum distance observed so far
  // and update min_distance
  if (calc_distance <= params->min_distance) {
    params->min_distance = calc_distance;

    // if smaller than last item in vector, remove that item first
    pairs_size = kv_size(*(params->dist_pairs));
    if (pairs_size > 0 &&
        calc_distance < (kv_A(*(params->dist_pairs), pairs_size - 1)).distance) {
      kv_pop(*(params->dist_pairs));
    }

    tree_geom_dist_vec_item_t dist_pair =
        (tree_geom_dist_vec_item_t){(GeometryObject**)item1, calc_distance};
    kv_push(tree_geom_dist_vec_item_t, *(params->dist_pairs), dist_pair);
  }

  // Set distance for callback
  // This adds a slight adjustment to force checking of adjacent tree nodes; otherwise
  // they are skipped by the GEOS nearest neighbor algorithm check against bounds
  // of adjacent nodes.
  *distance = calc_distance + 1e-6;

  return 1;
}

/* Find the nearest item in the tree to each input geometry.  Returns indices of
 * source array, tree items, and distance between them.
 *
 * If there are multiple equidistant or intersected items, all should be returned.
 * Tree indexes are returned in the order they are visited, not necessarily in ascending
 * order.
 *
 * Returns a tuple of empty arrays (shape (2,0), shape (0,)) if tree is empty.
 *
 *
 * Returns
 * -------
 * tuple of ([arr indexes (shape n), tree indexes (shape n)], distances (shape n))
 * */

#if GEOS_SINCE_3_6_0

static PyObject* STRtree_nearest(STRtreeObject* self, PyObject* arr) {
  PyArrayObject* pg_geoms;
  GeometryObject* pg_geom = NULL;
  GEOSGeometry* geom = NULL;
  GeometryObject** nearest_result = NULL;
  npy_intp i, n, size, geom_index;
  size_t j;
  char* head_ptr = (char*)self->_geoms;
  tree_nearest_userdata_t userdata;
  double distance;
  PyArrayObject* result_indexes;    // array of [source index, tree index]
  PyArrayObject* result_distances;  // array of distances
  PyObject* result;                 // tuple of (indexes array, distance array)

  // Indices of input geometries
  index_vec_t src_indexes;

  // Pairs of addresses in tree geometries and distances; used in userdata
  tree_geom_dist_vec_t dist_pairs;

  // Addresses in tree geometries (_geoms) that match tree
  tree_geom_vec_t nearest_geoms;

  // Distances of nearest items
  tree_dist_vec_t nearest_dist;

  if (self->ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
    return NULL;
  }
  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Not an ndarray");
    return NULL;
  }

  pg_geoms = (PyArrayObject*)arr;
  if (!PyArray_ISOBJECT(pg_geoms)) {
    PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
    return NULL;
  }

  if (PyArray_NDIM(pg_geoms) != 1) {
    PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
    return NULL;
  }

  // If tree is empty, return empty arrays
  if (self->count == 0) {
    npy_intp index_dims[2] = {2, 0};
    result_indexes = (PyArrayObject*)PyArray_SimpleNew(2, index_dims, NPY_INTP);

    npy_intp distance_dims[1] = {0};
    result_distances = (PyArrayObject*)PyArray_SimpleNew(1, distance_dims, NPY_DOUBLE);

    result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, (PyObject*)result_indexes);
    PyTuple_SET_ITEM(result, 1, (PyObject*)result_distances);
    return (PyObject*)result;
  }

  n = PyArray_SIZE(pg_geoms);

  // preallocate arrays to size of input array; for a non-empty tree, there should be
  // at least 1 nearest item per input geometry
  kv_init(src_indexes);
  kv_resize(npy_intp, src_indexes, n);
  kv_init(nearest_geoms);
  kv_resize(GeometryObject**, nearest_geoms, n);
  kv_init(nearest_dist);
  kv_resize(double, nearest_dist, n);

  GEOS_INIT_THREADS;

  // initialize userdata context and dist_pairs vector
  userdata.ctx = ctx;
  userdata.dist_pairs = &dist_pairs;

  for (i = 0; i < n; i++) {
    // get pygeos geometry from input geometry array
    pg_geom = *(GeometryObject**)PyArray_GETPTR1(pg_geoms, i);
    if (!get_geom(pg_geom, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      break;
    }
    if (geom == NULL || GEOSisEmpty_r(ctx, geom)) {
      continue;
    }

    // reset loop-dependent values of userdata
    kv_init(dist_pairs);
    userdata.min_distance = DBL_MAX;

    nearest_result = (GeometryObject**)GEOSSTRtree_nearest_generic_r(
        ctx, self->ptr, &pg_geom, geom, distance_callback, &userdata);

    // GEOSSTRtree_nearest_generic_r returns NULL on error
    if (nearest_result == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      kv_destroy(dist_pairs);
      kv_destroy(src_indexes);
      kv_destroy(nearest_geoms);
      kv_destroy(nearest_dist);
      break;
    }

    for (j = 0; j < kv_size(dist_pairs); j++) {
      distance = kv_A(dist_pairs, j).distance;

      // only keep entries from the smallest distances for this input geometry
      // Note: there may be multiple equidistant or intersected tree items
      if (distance <= userdata.min_distance) {
        kv_push(npy_intp, src_indexes, i);
        kv_push(GeometryObject**, nearest_geoms, kv_A(dist_pairs, j).geom);
        kv_push(double, nearest_dist, distance);
      }
    }

    kv_destroy(dist_pairs);
  }

  GEOS_FINISH_THREADS;

  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  size = kv_size(src_indexes);

  // Create array of indexes
  npy_intp index_dims[2] = {2, size};
  result_indexes = (PyArrayObject*)PyArray_SimpleNew(2, index_dims, NPY_INTP);
  if (result_indexes == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
    return NULL;
  }

  // Create array of distances
  npy_intp distance_dims[1] = {size};
  result_distances = (PyArrayObject*)PyArray_SimpleNew(1, distance_dims, NPY_DOUBLE);
  if (result_distances == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
    return NULL;
  }

  for (i = 0; i < size; i++) {
    // assign value into numpy arrays
    *(npy_intp*)PyArray_GETPTR2(result_indexes, 0, i) = kv_A(src_indexes, i);

    // Calculate index using offset of its address compared to head of _geoms
    geom_index =
        (npy_intp)(((char*)kv_A(nearest_geoms, i) - head_ptr) / sizeof(GeometryObject*));
    *(npy_intp*)PyArray_GETPTR2(result_indexes, 1, i) = geom_index;

    *(double*)PyArray_GETPTR1(result_distances, i) = kv_A(nearest_dist, i);
  }

  kv_destroy(src_indexes);
  kv_destroy(nearest_geoms);
  kv_destroy(nearest_dist);

  // return a tuple of (indexes array, distances array)
  result = PyTuple_New(2);
  PyTuple_SET_ITEM(result, 0, (PyObject*)result_indexes);
  PyTuple_SET_ITEM(result, 1, (PyObject*)result_distances);

  return (PyObject*)result;
}

#endif  // GEOS_SINCE_3_6_0

static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtreeObject, ptr), READONLY,
     "Pointer to GEOSSTRtree"},
    {"count", T_LONG, offsetof(STRtreeObject, count), READONLY,
     "The number of geometries inside the tree"},
    {NULL} /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
    {"query", (PyCFunction)STRtree_query, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search geometry, "
     "and optionally tests them "
     "against predicate function if provided. "},
    {"query_bulk", (PyCFunction)STRtree_query_bulk, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search "
     "geometries, and optionally tests them "
     "against predicate function if provided. "},
#if GEOS_SINCE_3_6_0
    {"nearest", (PyCFunction)STRtree_nearest, METH_O,
     "Queries the index for the nearest item to each of the given search geometries"},
#endif
    {NULL} /* Sentinel */
};

PyTypeObject STRtreeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygeos.lib.STRtree",
    .tp_doc =
        "A query-only R-tree created using the Sort-Tile-Recursive (STR) algorithm.",
    .tp_basicsize = sizeof(STRtreeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = STRtree_new,
    .tp_dealloc = (destructor)STRtree_dealloc,
    .tp_members = STRtree_members,
    .tp_methods = STRtree_methods};

int init_strtree_type(PyObject* m) {
  if (PyType_Ready(&STRtreeType) < 0) {
    return -1;
  }

  Py_INCREF(&STRtreeType);
  PyModule_AddObject(m, "STRtree", (PyObject*)&STRtreeType);
  return 0;
}
