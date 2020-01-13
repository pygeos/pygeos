from enum import IntEnum
import numpy as np
from pygeos import lib


__all__ = ["STRtree"]


class UnaryPredicate(IntEnum):
    """The enumeration of GEOS unary predicates types"""

    intersects = 1
    within = 2
    contains = 3
    overlaps = 4
    crosses = 5
    touches = 6


VALID_PREDICATES = {e.name for e in UnaryPredicate}


class STRtree:
    """A query-only R-tree created using the Sort-Tile-Recursive (STR)
    algorithm.

    For two-dimensional spatial data. The actual tree will be constructed at the first
    query.

    Parameters
    ----------
    geometries : array_like
    leafsize : int
        the maximum number of child nodes that a node can have

    Examples
    --------
    >>> import pygeos
    >>> geoms = pygeos.points(np.arange(10), np.arange(10))
    >>> tree = pygeos.STRtree(geoms)
    >>> tree.query(pygeos.box(2, 2, 4, 4)).tolist()
    [2, 3, 4]
    """

    def __init__(self, geometries, leafsize=5):
        self._tree = lib.STRtree(np.asarray(geometries, dtype=np.object), leafsize)

    def __len__(self):
        return self._tree.count

    def query(self, geometry, predicate=None):
        """Return all items whose extent intersect the envelope of the input
        geometry.  If predicate is provided, these items are limited to those
        that satisfy the predicate operation when compared against the input
        geometry.

        If geometry is None, an empty array is returned.

        Parameters
        ----------
        geometry : Geometry
            The envelope of the geometry is taken automatically for
            querying the tree.
        predicate : str, optional (default: None)
            The predicate to use for testing geometries from the tree
            that are within the input geometry's envelope.
        """

        if geometry is None:
            return np.array([], dtype="int")

        if predicate is None:
            predicate = 0

        else:
            if not predicate in VALID_PREDICATES:
                raise ValueError(
                    "Predicate {} is not valid; must be one of {}".format(
                        predicate, ", ".join(VALID_PREDICATES)
                    )
                )

            predicate = UnaryPredicate[predicate].value

        return self._tree.query(geometry, predicate)

    @property
    def geometries(self):
        return self._tree.geometries
