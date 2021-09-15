import pygeos

empty_point = pygeos.Geometry("POINT EMPTY")
empty_point_z = pygeos.Geometry("POINT Z EMPTY")


def test_coordinate_dimension_2d():
    assert pygeos.get_coordinate_dimension(empty_point) == 2


def test_coordinate_dimension_3d():
    assert pygeos.get_coordinate_dimension(empty_point_z) == 2


def test_has_z_2d():
    assert not pygeos.has_z(empty_point)


def test_has_z_3d():
    assert pygeos.has_z(empty_point_z)


def test_wkt_2d():
    assert pygeos.to_wkt(empty_point) == "POINT EMPTY"


def test_wkt_3d():
    assert pygeos.to_wkt(empty_point_z) == "POINT Z EMPTY"


def test_wkb_2d():
    assert pygeos.to_wkb(empty_point, hex=True) == '0101000000000000000000F87F000000000000F87F'


def test_wkb_3d():
    assert pygeos.to_wkb(empty_point_z, hex=True) == '0101000080000000000000F87F000000000000F87F000000000000F87F'
