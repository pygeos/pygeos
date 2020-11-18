# distutils: define_macros=GEOS_USE_ONLY_R_API

from libc.stdlib cimport free, malloc, realloc
from cpython cimport Py_buffer
cimport cython
cimport numpy as np
import numpy as np

from pygeos._geos cimport GEOSGeometry, GEOSContextHandle_t
from pygeos._pygeos_api cimport import_pygeos_c_api, PyGEOS_CreateGeometry


# initialize PyGEOS C API
import_pygeos_c_api()


@cython.final
cdef class GeometryVector:
    def __cinit__(self, Py_ssize_t reserved_size=0):
        if reserved_size < 0:
            raise ValueError("reserved_size must be >= 0")

        self._reserved_size = reserved_size
        self._size = 0

        if reserved_size:
            self._values = <GEOSGeometry**>malloc(sizeof(GEOSGeometry*) * reserved_size)
            if self._values == NULL:
                raise MemoryError
        else:
            self._values = NULL

    def __dealloc__(self):
        self.clear()

    cdef GeometryVector __enter__(self, Py_ssize_t reserved_size=0):
        return GeometryVector(reserved_size)

    def __exit__(self, type, value, traceback):
        self.clear()

    def __len__(self):
        return self._size

    cdef void clear(self) nogil:
        self._size = 0
        self._reserved_size = 0
        if self._values != NULL:
            # TODO: potential memory leak, do we need to release GEOS geoms?
            # Probably need instance-level property to know if they are owned by this
            # vector
            free(self._values)
            self._values = NULL

    cdef GEOSGeometry* get(self, Py_ssize_t index) nogil:
        # WARNING: this will likely raise segfaults for access outside allocated values

        # support negative indexing
        if index < 0:
            index = self._size + index

        return self._values[index]

    cdef void push(self, GEOSGeometry *value) nogil:
        if self._size == self._reserved_size:
            # no more reserved availabe, reallocate 2x reserved_size, minimum of 2
            self._reserved_size = self._reserved_size * 2 or 2

            next_ptr = <GEOSGeometry**>realloc(self._values,
                                                sizeof(GEOSGeometry*) * self._reserved_size)
            if next_ptr == NULL:
                # cleanup any allocated memory first
                self.clear()
                raise MemoryError

            self._values = next_ptr

        self._values[self._size] = value
        self._size += 1

    cdef Py_ssize_t size(self) nogil:
        return self._size

    # NOTE: returns GeometryObjects not GEOSGeometry
    # corresponding view didn't work because views of pointer types not yet supported
    cdef to_array(self, GEOSContextHandle_t geos_handle):
        cdef Py_ssize_t i = 0

        out = np.empty(shape=(self._size, ), dtype=np.object)
        cdef object[:] out_view = out

        for i in range(self._size):
            out_view[i] = PyGEOS_CreateGeometry(<GEOSGeometry *>self._values[i], geos_handle)

        return out


@cython.final
cdef class IndexVector:
    def __cinit__(self, Py_ssize_t reserved_size=0):
        if reserved_size < 0:
            raise ValueError("reserved_size must be >= 0")

        self._reserved_size = reserved_size
        self._size = 0

        if reserved_size:
            self._values = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * reserved_size)
            if self._values == NULL:
                raise MemoryError
        else:
            self._values = NULL

    def __dealloc__(self):
        self.clear()

    cdef IndexVector __enter__(self, Py_ssize_t reserved_size=0):
        return IndexVector(reserved_size)

    def __exit__(self, type, value, traceback):
        self.clear()

    def __getitem__(self, index):
        # check for valid bounds (TODO: requires gil)
        # if index < 0 or index >= self._size:
        #     raise ValueError("index is outside bounds of vector")

        return self.get(index)

    def __len__(self):
        return self._size

    cdef void clear(self) nogil:
        self._size = 0
        self._reserved_size = 0
        if self._values != NULL:
            free(self._values)
            self._values = NULL

    cdef Py_ssize_t get(self, Py_ssize_t index) nogil:
        # WARNING: this will likely raise segfaults for access outside allocated values

        # support negative indexing
        if index < 0:
            index = self._size + index

        return self._values[index]

    cdef void push(self, Py_ssize_t value) nogil:
        if self._size == self._reserved_size:
            # no more reserved availabe, reallocate 2x reserved_size, minimum of 2
            self._reserved_size = self._reserved_size * 2 or 2

            next_ptr = <Py_ssize_t*>realloc(self._values,
                                                sizeof(Py_ssize_t) * self._reserved_size)
            if next_ptr == NULL:
                # cleanup any allocated memory first
                self.clear()
                raise MemoryError

            self._values = next_ptr

        self._values[self._size] = value
        self._size += 1

    def to_array(self):
        if self._size == 0:
            return np.empty(shape=(0,), dtype=np.intp)

        cdef Py_ssize_t[:] view = <Py_ssize_t[:self._size]>self._values
        return np.array(view)

    cdef Py_ssize_t[:] get_view(self):
        cdef Py_ssize_t[:] view = <Py_ssize_t[:self._size]>self._values
        return view

    cdef Py_ssize_t size(self) nogil:
        return self._size


