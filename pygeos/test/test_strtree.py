import pygeos
import numpy as np
from .common import point


def test_init():
    arr = np.array([point])
    tree = pygeos.lib.STRtree(arr, 5)
    assert tree.geometries is arr
    del tree
    del arr
