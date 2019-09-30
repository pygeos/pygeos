from . import ufuncs, Geometry
import numpy as np

__all__ = ["apply", "count_coordinates", "get_coordinates", "set_coordinates"]


def apply(geometry, transformation):
    """Apply a function to the coordinates of a geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    transformation : function
        A function that transforms a (N, 2) ndarray of float64 to another
        (N, 2) ndarray of float64.

    Examples
    --------
    >>> apply(Geometry("POINT (0 0)"), lambda x: x + 1)
    <pygeos.Geometry POINT (1 1)>
    >>> apply(Geometry("LINESTRING (2 2, 4 4)"), lambda x: x * [2, 3])
    <pygeos.Geometry LINESTRING (4 6, 8 12)>
    >>> apply(None, lambda x: x) is None
    True
    >>> apply([Geometry("POINT (0 0)"), None], lambda x: x).tolist()
    [<pygeos.Geometry POINT (0 0)>, None]
    """
    geometry = np.asarray(geometry, dtype=np.object)
    coordinates = ufuncs.get_coordinates(geometry)
    coordinates = transformation(coordinates)
    geometry = ufuncs.set_coordinates(geometry, coordinates)
    if geometry.ndim == 0:
        return geometry.item()
    return geometry


def count_coordinates(geometry):
    """Count the number of coordinate pairs in a geometry array

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> count_coordinates(Geometry("POINT (0 0)"))
    1
    >>> count_coordinates(Geometry("LINESTRING (2 2, 4 4)"))
    2
    >>> count_coordinates(None)
    0
    >>> count_coordinates([Geometry("POINT (0 0)"), None])
    1
    """
    return ufuncs.count_coordinates(np.asarray(geometry, dtype=np.object))


def get_coordinates(geometry):
    """Get coordinates from a geometry array as a float array

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> get_coordinates(Geometry("POINT (0 0)"))
    array([[0., 0.]])
    >>> get_coordinates(Geometry("LINESTRING (2 2, 4 4)"))
    array([[2., 2.],
           [4., 4.]])
    >>> get_coordinates(None)
    array([], shape=(0, 2), dtype=float64)
    >>> get_coordinates([Geometry("POINT (0 0)"), None])
    array([[0., 0.]])
    """
    return ufuncs.get_coordinates(np.asarray(geometry, dtype=np.object))


def set_coordinates(geometry, coordinates):
    """Returns a copy of a geometry array with different coordinates.

    Parameters
    ----------
    geometry : Geometry or array_like
    coordinates: array_like

    Examples
    --------
    >>> set_coordinates(Geometry("POINT (0 0)"), [[1, 1]])
    <pygeos.Geometry POINT (1 1)>
    >>> set_coordinates([Geometry("POINT (0 0)"), Geometry("LINESTRING (0 0, 0 0)")], [[1, 2], [3, 4], [5, 6]]).tolist()
    [<pygeos.Geometry POINT (1 2)>, <pygeos.Geometry LINESTRING (3 4, 5 6)>]
    >>> set_coordinates([None, Geometry("POINT (0 0)")], [[1, 2]]).tolist()
    [None, <pygeos.Geometry POINT (1 2)>]
    """
    geometry = np.array(geometry, dtype=np.object)  # makes a copy
    coordinates = np.atleast_2d(np.asarray(coordinates)).astype(np.float64)
    if coordinates.shape != (ufuncs.count_coordinates(geometry), 2):
        raise ValueError(
            "The coordinate array has an invalid shape {}".format(coordinates.shape)
        )
    ufuncs.set_coordinates(geometry, coordinates)
    if geometry.ndim == 0:
        return geometry.item()
    return geometry
