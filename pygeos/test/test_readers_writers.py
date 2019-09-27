import numpy
import pygeos
import pytest

from pygeos.ufuncs import from_wkt, from_wkb


def test_from_wkt():

    expected = pygeos.points(1, 1)
    actual = from_wkt('POINT (1 1)')
    assert pygeos.equals(actual, expected)
    actual = from_wkt(b'POINT (1 1)')
    assert pygeos.equals(actual, expected)
    
    with pytest.raises(TypeError, match="Expected bytes, found int"):
        from_wkt(1)
    
    with pytest.raises(pygeos.GEOSException):
        from_wkt('invalid')


def test_from_wkb():

    expected = pygeos.points(1, 1)
    actual = from_wkb(b'\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?')
    assert pygeos.equals(actual, expected)
    # HEX form
    actual = from_wkb(b'0101000000000000000000F03F000000000000F03F')
    assert pygeos.equals(actual, expected)

    with pytest.raises(TypeError, match="Expected bytes, found str"):
        from_wkb("test")

    with pytest.raises(pygeos.GEOSException):
        from_wkb(b'\x01\x01\x00\x00\x00\x00')
