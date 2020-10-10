import numpy as np
import pygeos

from .common import point, multi_point, geometry_collection

def test_cythondemo():
    from pygeos.cythondemo import geos_get_num_geometries
    arr = np.array([point, multi_point, geometry_collection])
    assert geos_get_num_geometries(arr).tolist() == [1, 2, 2]
