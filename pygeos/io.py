import numpy as np

from . import Geometry  # noqa
from . import lib


__all__ = ["from_wkb", "from_wkt", "to_wkb", "to_wkt"]


def to_wkt(
    geometry,
    rounding_precision=6,
    trim=True,
    output_dimension=3,
    old_3d=False,
    **kwargs
):
    """
    Converts to the Well-Known Text (WKT) representation of a Geometry.

    The Well-known Text format is defined in the `OGC Simple Features
    Specification for SQL <https://www.opengeospatial.org/standards/sfs>`__.

    Parameters
    ----------
    geometry : Geometry or array_like
    rounding_precision : int, default 6
        The rounding precision when writing the WKT string. Set to a value of
        -1 to indicate the full precision.
    trim : bool, default True
        Whether to trim unnecessary decimals (trailing zeros).
    output_dimension : int, default 3
        The output dimension for the WKT string. Supported values are 2 and 3.
        Specifying 3 means that up to 3 dimensions will be written but 2D
        geometries will still be represented as 2D in the WKT string.
    old_3d : bool, default False
        Enable old style 3D/4D WKT generation. By default, new style 3D/4D WKT
        (ie. "POINT Z (10 20 30)") is returned, but with ``old_3d=True``
        the WKT will be formatted in the style "POINT (10 20 30)".

    Examples
    --------
    >>> to_wkt(Geometry("POINT (0 0)"))
    'POINT (0 0)'
    >>> to_wkt(Geometry("POINT (0 0)"), rounding_precision=3, trim=False)
    'POINT (0.000 0.000)'
    >>> to_wkt(Geometry("POINT (0 0)"), rounding_precision=-1, trim=False)
    'POINT (0.0000000000000000 0.0000000000000000)'
    >>> to_wkt(Geometry("POINT (1 2 3)"), trim=True)
    'POINT Z (1 2 3)'
    >>> to_wkt(Geometry("POINT (1 2 3)"), trim=True, output_dimension=2)
    'POINT (1 2)'
    >>> to_wkt(Geometry("POINT (1 2 3)"), trim=True, old_3d=True)
    'POINT (1 2 3)'

    Notes
    -----
    The defaults differ from the default of the GEOS library. To mimic this,
    use::

        to_wkt(geometry, rounding_precision=-1, trim=False, output_dimension=2)

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


def to_wkb(geometry, hex=False, output_dimension=3, byte_order=-1, include_srid=False, **kwargs):
    r"""
    Converts to the Well-Known Binary (WKB) representation of a Geometry.

    The Well-Known Binary format is defined in the `OGC Simple Features
    Specification for SQL <https://www.opengeospatial.org/standards/sfs>`__.

    Parameters
    ----------
    geometry : Geometry or array_like
    hex : bool, default False
        If true, export the WKB as a hexidecimal string. The default is to
        return a binary bytes object.
    output_dimension : int, default 3
        The output dimension for the WKB. Supported values are 2 and 3.
        Specifying 3 means that up to 3 dimensions will be written but 2D
        geometries will still be represented as 2D in the WKB represenation.
    byte_order : int
        Defaults to native machine byte order (-1). Use 0 to force big endian
        and 1 for little endian.
    include_srid : bool, default False
        Whether the SRID should be included in WKB (this is an extension
        to the OGC WKB specification).

    Examples
    --------
    >>> to_wkb(Geometry("POINT (1 1)"))
    b'\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?'
    >>> to_wkb(Geometry("POINT (1 1)"), hex=True)
    '0101000000000000000000F03F000000000000F03F'
    """
    if not np.isscalar(hex):
        raise TypeError("hex only accepts scalar values")
    if not np.isscalar(output_dimension):
        raise TypeError("output_dimension only accepts scalar values")
    if not np.isscalar(byte_order):
        raise TypeError("byte_order only accepts scalar values")
    if not np.isscalar(include_srid):
        raise TypeError("include_srid only accepts scalar values")

    return lib.to_wkb(
        geometry,
        np.bool(hex),
        np.intc(output_dimension),
        np.intc(byte_order),
        np.bool(include_srid),
        **kwargs,
    )


def from_wkt(geometry, **kwargs):
    """
    Creates geometries from the Well-Known Text (WKT) representation.

    The Well-known Text format is defined in the `OGC Simple Features
    Specification for SQL <https://www.opengeospatial.org/standards/sfs>`__.

    Parameters
    ----------
    geometry : str or array_like
        The WKT string(s) to convert.

    Examples
    --------
    >>> from_wkt('POINT (0 0)')
    <pygeos.Geometry POINT (0 0)>
    """
    return lib.from_wkt(geometry, **kwargs)


def from_wkb(geometry, **kwargs):
    r"""
    Creates geometries from the Well-Known Binary (WKB) representation.

    The Well-Known Binary format is defined in the `OGC Simple Features
    Specification for SQL <https://www.opengeospatial.org/standards/sfs>`__.

    Parameters
    ----------
    geometry : str or array_like
        The WKB byte object(s) to convert.

    Examples
    --------
    >>> from_wkb(b'\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?')
    <pygeos.Geometry POINT (1 1)>
    """
    # ensure the input has object dtype, to avoid numpy inferring it as a
    # fixed-length string dtype (which removes trailing null bytes upon access
    # of array elements)
    geometry = np.asarray(geometry, dtype=object)
    return lib.from_wkb(geometry, **kwargs)
