import numpy as np
from numpy.testing import assert_almost_equal

import pygeos

__all__ = ["assert_geometries_equal"]


def _assert_nan_coords_same(x, y, decimal):
    x_ndim = pygeos.get_coordinate_dimension(x)
    y_ndim = pygeos.get_coordinate_dimension(y)
    if x_ndim != y_ndim:
        return False

    x_coords = pygeos.get_coordinates(x, include_z=x_ndim == 3)
    y_coords = pygeos.get_coordinates(y, include_z=x_ndim == 3)

    # Check NaN equality
    try:
        assert_almost_equal(x_coords, y_coords)
    except AssertionError:
        return False
    else:
        return True


def _assert_none_same(x, y):
    x_id = pygeos.is_missing(x)
    y_id = pygeos.is_missing(y)

    if not (x_id == y_id).all():
        raise AssertionError(
            "One of the arrays contains a None where the other has a geometry."
        )

    # If there is a scalar, then here we know the array has the same
    # flag as it everywhere, so we should return the scalar flag.
    if isinstance(x_id, bool) or x_id.ndim == 0:
        return bool(x_id)
    elif isinstance(y_id, bool) or y_id.ndim == 0:
        return bool(y_id)
    else:
        return y_id


def assert_geometries_equal(x, y, decimal=7, equal_none=True, equal_nan=True):
    """Raises an AssertionError if two geometry array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all elements of
    these objects are equal. An exception is raised at shape mismatch or conflicting
    values. In contrast to the standard usage in pygeos, NaNs and Nones are compared like numbers,
    no assertion is raised if both objects have NaNs/Nones in the same positions.
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    # Check the shapes (condition is copied from numpy test_array_equal)
    if not ((x.shape == () or y.shape == ()) or x.shape == y.shape):
        raise AssertionError(
            f"Arrays not equal: (shapes {x.shape}, {y.shape} mismatch)"
        )

    if (not pygeos.is_valid_input(x).all()) or (not pygeos.is_valid_input(y).all()):
        raise AssertionError("One of the arrays contains non-geometry input.")

    flagged = False
    if equal_none:
        flagged = _assert_none_same(x, y)

    if flagged.ndim > 0:
        x, y = x[~flagged], y[~flagged]
        # Only do the comparison if actual values are left
        if x.size == 0:
            return
    elif flagged:
        # no sense doing comparison if everything is flagged.
        return

    is_equal = pygeos.equals_exact(x, y, tolerance=10 ** -decimal)

    if not equal_nan:
        return
    if is_equal.ndim > 0:
        x, y = x[~is_equal], y[~is_equal]
        # Only do the NaN check if actual values are left
        if x.size == 0:
            return
    elif is_equal:
        # no sense in checking for NaN if everything is equal.
        return

    for (_x, _y) in np.broadcast(x, y):
        if not _assert_nan_coords_same(_x, _y):
            raise AssertionError("One of the geometries are not equal")
