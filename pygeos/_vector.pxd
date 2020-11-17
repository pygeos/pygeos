cimport numpy as np
from pygeos._geos cimport GEOSGeometry, GEOSContextHandle_t


cdef class GeometryVector:
    cdef Py_ssize_t _reserved_size
    cdef Py_ssize_t _size
    cdef GEOSGeometry** _values
    cdef GeometryVector __enter__(self, Py_ssize_t reserved_size=?)
    cdef void clear(self) nogil
    cdef GEOSGeometry* get(self, Py_ssize_t index) nogil
    cdef void push(self, GEOSGeometry *value) nogil
    cdef Py_ssize_t size(self) nogil
    cdef to_array(self, GEOSContextHandle_t geos_handle)


cdef class IndexVector:
    cdef Py_ssize_t _reserved_size
    cdef Py_ssize_t _size
    cdef Py_ssize_t* _values
    cdef IndexVector __enter__(self, Py_ssize_t reserved_size=?)
    cdef void clear(self) nogil
    cdef Py_ssize_t get(self, Py_ssize_t index) nogil
    cdef void push(self, Py_ssize_t value) nogil
    cdef Py_ssize_t size(self) nogil
    cdef Py_ssize_t[:] get_view(self)
















# from pygeos._geos cimport GEOSGeometry

# cdef extern from "kvec.h":
#     size_t kv_size(...)
#     void kv_init(...)



# cdef extern from "vector.h":
#     ctypedef struct index_vec_t:
#         pass
#         # size_t n
#         # size_t m
#         # Py_ssize_t* a;

#     # ctypedef index_vec index_vec_t

#     cdef index_vec_t index_vec_init(index_vec_t vector) nogil
#     cdef void index_vec_destroy(index_vec_t vector) nogil
#     cdef Py_ssize_t index_vec_get(index_vec_t vector, size_t index) nogil
#     cdef void index_vec_push(index_vec_t vector, Py_ssize_t value) nogil

#     cdef struct geom_vec:
#         size_t n
#         size_t m
#         GEOSGeometry** a;

#     ctypedef geom_vec geom_vec_t

#     cdef geom_vec_t geom_vec_create()
#     cdef void geom_vec_destroy(geom_vec_t vector) nogil
#     cdef GEOSGeometry geom_vec_get(geom_vec_t vector, size_t index) nogil
#     cdef void geom_vec_push(geom_vec_t vector, GEOSGeometry *value) nogil
#     size_t geom_vec_size(geom_vec_t vector) nogil
