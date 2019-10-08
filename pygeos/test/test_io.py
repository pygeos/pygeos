import numpy
import pygeos
import pytest


POINT11_WKB = (
    b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"
)


def test_from_wkt():
    expected = pygeos.points(1, 1)
    actual = pygeos.from_wkt("POINT (1 1)")
    assert pygeos.equals(actual, expected)
    actual = pygeos.from_wkt(b"POINT (1 1)")
    assert pygeos.equals(actual, expected)

    # None propagates
    assert pygeos.from_wkt(None) is None

    with pytest.raises(TypeError, match="Expected bytes, got int"):
        pygeos.from_wkt(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.from_wkt("invalid")


def test_from_wkb():
    expected = pygeos.points(1, 1)
    actual = pygeos.from_wkb(POINT11_WKB)
    assert pygeos.equals(actual, expected)
    # HEX form
    actual = pygeos.from_wkb(b"0101000000000000000000F03F000000000000F03F")
    assert pygeos.equals(actual, expected)

    # None propagates
    assert pygeos.from_wkb(None) is None

    with pytest.raises(TypeError, match="Expected bytes, got str"):
        pygeos.from_wkb("test")

    with pytest.raises(pygeos.GEOSException):
        pygeos.from_wkb(b"\x01\x01\x00\x00\x00\x00")


def test_to_wkt():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkt(point)
    assert actual == "POINT (1 1)"

    actual = pygeos.to_wkt(point, rounding_precision=2, trim=False)
    assert actual == "POINT (1.00 1.00)"

    # 3D points
    point_z = pygeos.points(1, 1, 1)
    actual = pygeos.to_wkt(point_z)
    assert actual == "POINT Z (1 1 1)"
    actual = pygeos.to_wkt(point_z, output_dimension=3)
    assert actual == "POINT Z (1 1 1)"

    actual = pygeos.to_wkt(point_z, output_dimension=2)
    assert actual == "POINT (1 1)"

    actual = pygeos.to_wkt(point_z, old_3d=True)
    assert actual == "POINT (1 1 1)"

    # None propagates
    assert pygeos.to_wkt(None) is None

    with pytest.raises(TypeError):
        pygeos.to_wkt(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.to_wkt(point, output_dimension=4)


def test_to_wkb():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkb(point)
    assert actual == POINT11_WKB

    actual = pygeos.to_wkb(point, hex=True)
    assert actual == '0101000000000000000000F03F000000000000F03F'

    point_z = pygeos.points(1, 1, 1)
    actual = pygeos.to_wkb(point_z)
    assert actual == b'\x01\x01\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?'  # noqa
    actual = pygeos.to_wkb(point_z, output_dimension=2)
    assert actual == POINT11_WKB

    # None propagates
    assert pygeos.to_wkb(None) is None

    with pytest.raises(TypeError):
        pygeos.to_wkb(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.to_wkb(point, output_dimension=4)


def test_srid_roundtrip():
    # hex representation of POINT (0 0) with SRID=4
    ewkb = "01010000200400000000000000000000000000000000000000"
    wkb = "010100000000000000000000000000000000000000"

    actual = pygeos.from_wkb(ewkb.encode())
    assert pygeos.to_wkt(actual, trim=True) == 'POINT (0 0)'

    assert pygeos.to_wkb(actual, hex=True) == wkb
    assert pygeos.to_wkb(actual, hex=True, include_srid=True) == ewkb
