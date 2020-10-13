cdef GEOSContextHandle_t get_geos_handle():
    """Provides GEOS context pointer to use in GEOS functions called from
    within Cython functions.
    """

    cdef GEOSContextHandle_t handle

    handle = GEOS_init_r()
    return handle
