import numpy
import pygeos
import pytest

from pygeos.lib import from_wkt, from_wkb, to_wkb
from pygeos.io import to_wkt


POINT11_WKB = (
    b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"
)


def test_from_wkt():
    expected = pygeos.points(1, 1)
    actual = from_wkt("POINT (1 1)")
    assert pygeos.equals(actual, expected)
    actual = from_wkt(b"POINT (1 1)")
    assert pygeos.equals(actual, expected)

    # None propagates
    assert from_wkb(None) is None

    with pytest.raises(TypeError, match="Expected bytes, got int"):
        from_wkt(1)

    with pytest.raises(pygeos.GEOSException):
        from_wkt("invalid")


def test_from_wkb():
    expected = pygeos.points(1, 1)
    actual = from_wkb(POINT11_WKB)
    assert pygeos.equals(actual, expected)
    # HEX form
    actual = from_wkb(b"0101000000000000000000F03F000000000000F03F")
    assert pygeos.equals(actual, expected)

    # None propagates
    assert from_wkb(None) is None

    with pytest.raises(TypeError, match="Expected bytes, got str"):
        from_wkb("test")

    with pytest.raises(pygeos.GEOSException):
        from_wkb(b"\x01\x01\x00\x00\x00\x00")


def test_to_wkt():
    points = pygeos.points(1, 1, 1)
    actual = to_wkt(points, trim=True)
    assert actual == "POINT (1 1)"

    actual = to_wkt(points, rounding_precision=2)
    assert actual == "POINT (1.00 1.00)"

    actual = to_wkt(points, trim=True, output_dimension=3)
    assert actual == "POINT Z (1 1 1)"

    actual = to_wkt(points, trim=True, output_dimension=3, old_3d=True)
    assert actual == "POINT (1 1 1)"

    # None propagates
    assert to_wkt(None) is None

    with pytest.raises(TypeError):
        to_wkt(1)

    with pytest.raises(pygeos.GEOSException):
        to_wkt(points, output_dimension=4)


def test_to_wkb():
    points = pygeos.points(1, 1)
    actual = to_wkb(points)
    assert actual == POINT11_WKB

    # None propagates
    assert to_wkb(None) is None

    with pytest.raises(TypeError):
        to_wkb(1)

