from . import ufuncs, Geometry
import numpy as np

__all__ = ["apply", "transform"]


def apply(geometries, transformation):
    """Apply a function to the coordinates of a geometry.

    Parameters
    ----------
    geometries : Geometry or array_like
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
    geometries = np.asarray(geometries, dtype=np.object)
    coordinates = ufuncs.get_coordinates(geometries)
    coordinates = transformation(coordinates)
    geometries = ufuncs.set_coordinates(geometries, coordinates)
    if geometries.ndim == 0:
        return geometries.item()
    return geometries


def transform(geometries, crs_from, crs_to):
    """Transform the coordinates from one CRS to another.

    Parameters
    ----------
    geometries : Geometry or array_like
    crs_from : CRS or input used to create one
        Projection of input data.
    crs_to : CRS or input used to create one
        Projection of output data.

    See also
    --------
    https://pyproj4.github.io

    Examples
    --------
    >>> transform(Geometry("POINT (136687 455783)"), 28992, 4326)
    <pygeos.Geometry POINT (52.1 5.12)>
    """
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("This function requires pyproj >= 2.1.0")
    transformer = Transformer.from_crs(crs_from, crs_to)

    # pyproj takes a (2, N) array, while we have (N, 2)
    def transformation(coords):
        x, y = transformer.transform(*coords.T)
        return np.array([x, y]).T

    return apply(geometries, transformation)
