import numpy as np
from . import Geometry  # noqa
from . import lib


__all__ = ["from_wkb", "from_wkt", "to_wkb", "to_wkb"]


def to_wkt(
    geometry,
    rounding_precision=-1,
    trim=False,
    output_dimension=2,
    old_3d=False,
    **kwargs
):
    """
    Converts to the Well-Known Text (WKT) representation of a  Geometry.

    The Well-known Text format is defined in the `OGC Simple Features
    Specification for SQL <http://www.opengis.org/techno/specs.htm>`__.

    Parameters
    ----------
    geometry : Geometry or array_like
    rounding_precision : int
        The rounding precision when writing the WKT string. A value of -1
        indicates the full precision.
    trim : bool
        Whether to trim unnecessary decimals (trailing zero's).
    output_dimension : int
        The output dimension for the WKT string. Supported values are 2 and 3.
        Specifying 3 means that up to 3 dimensions will be written but 2D
        geometries will still be represented as 2D in the WKT string.
    old_3d : bool
        Enable old style 3D/4D WKT generation. By default, new style 3D/4D WKT
        (ie. "POINT Z (10 20 30)") is returned, but with ``old_3d=True``
        the WKT will be formatted in the style "POINT (10 20 30)".

    Examples
    --------
    >>> to_wkt(Geometry("POINT (0 0)"))
    'POINT (0.0000000000000000 0.0000000000000000)'
    >>> to_wkt(Geometry("POINT (0 0)"), trim=True)
    'POINT (0 0)'
    >>> to_wkt(Geometry("POINT (0 0)"), rounding_precision=3)
    'POINT (0.000 0.000)'
    >>> to_wkt(Geometry("POINT (1 2 3)"), trim=True)
    'POINT (1 2)'
    >>> to_wkt(Geometry("POINT (1 2 3)"), trim=True, output_dimension=3)
    'POINT Z (1 2 3)'
    >>> to_wkt(Geometry("POINT (1 2 3)"), trim=True, output_dimension=3, old_3d=True)
    'POINT (1 2 3)'

    """
    if not np.isscalar(rounding_precision):
        raise TypeError("rounding_precision only accepts scalar values")
    if not np.isscalar(trim):
        raise TypeError("trim only accepts scalar values")
    if not np.isscalar(output_dimension):
        raise TypeError("output_dimension only accepts scalar values")
    if not np.isscalar(old_3d):
        raise TypeError("old_3d only accepts scalar values")

    return lib.to_wkt(
        geometry,
        np.intc(rounding_precision),
        np.bool(trim),
        np.intc(output_dimension),
        np.bool(old_3d),
        **kwargs,
    )
