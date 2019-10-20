from pygeos import lib
import numpy as np

__all__ = ["STRtree"]


class STRtree:
    """A query-only R-tree created using the Sort-Tile-Recursive (STR)
    algorithm.

    For two-dimensional spatial data.
    """

    def __init__(self, geometries, leafsize=5):
        self._tree = lib.STRtree(np.asarray(geometries, dtype=np.object), leafsize)

    def query(self, envelope):
        return self._tree.query(envelope)
