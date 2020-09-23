from pygeos.ext.geos cimport get_geos_handle
import pygeos as pg
import numpy as np

# TODO:
# @cython.boundscheck(False)
# @cython.wraparound(False)
def get_parts(array):

    input_index = np.arange(0, len(array))

    parts = []
    index = []
    for i in input_index:
        num_parts = pg.get_num_geometries(array[i])
        parts.extend(pg.get_geometry(array[i], range(num_parts)))
        index.extend(np.repeat(input_index[i], num_parts))

    return parts, index