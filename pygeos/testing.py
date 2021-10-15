import numpy as np

import pygeos

__all__ = ["assert_geometries_equal"]


def _equals_exact_with_ndim(x, y, tolerance):
    return pygeos.equals_exact(x, y, tolerance=tolerance) & (
        pygeos.get_coordinate_dimension(x) == pygeos.get_coordinate_dimension(y)
    )


def _replace_nan(arr):
    return np.where(np.isnan(arr), 0.0, arr)


def _assert_nan_coords_same(x, y, tolerance):
    x_coords = pygeos.get_coordinates(x, include_z=True)
    y_coords = pygeos.get_coordinates(y, include_z=True)

    # Check the shapes (condition is copied from numpy test_array_equal)
    if x_coords.shape != y_coords.shape:
        raise AssertionError(
            f"Coordinate arrays not equal: (shapes {x_coords.shape}, {y_coords.shape} mismatch)"
        )

    # Check NaN positional equality
    x_id = np.isnan(x_coords)
    y_id = np.isnan(y_coords)
    if not (x_id == y_id).all():
        raise AssertionError(
            "One of the geometries contains a NaN coordinate where the other does not."
        )

    # If this passed, replace NaN with a number to be able to use equals_exact
    x_no_nan = pygeos.apply(x, _replace_nan, include_z=True)
    y_no_nan = pygeos.apply(y, _replace_nan, include_z=True)

    return _equals_exact_with_ndim(x_no_nan, y_no_nan, tolerance=tolerance)


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


def assert_geometries_equal(x, y, tolerance=1e-7, equal_none=True, equal_nan=True):
    """Raises an AssertionError if two geometry array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all elements of
    these objects are equal. An exception is raised at shape mismatch or conflicting
    values. In contrast to the standard usage in pygeos, NaNs and Nones are compared like numbers,
    no assertion is raised if both objects have NaNs/Nones in the same positions.
    """
    # __tracebackhide__ = True  # Hide traceback for py.test
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

    if not np.isscalar(flagged):
        x, y = x[~flagged], y[~flagged]
        # Only do the comparison if actual values are left
        if x.size == 0:
            return
    elif flagged:
        # no sense doing comparison if everything is flagged.
        return

    is_equal = _equals_exact_with_ndim(x, y, tolerance=tolerance)
    if np.all(is_equal):
        return
    elif not equal_nan:
        raise AssertionError("One of the geometries are not equal")

    # Optionally refine failing elements if NaN should be considered equal
    if not np.isscalar(is_equal):
        x, y = x[~is_equal], y[~is_equal]
        # Only do the NaN check if actual values are left
        if x.size == 0:
            return
    elif is_equal:
        # no sense in checking for NaN if everything is equal.
        return

    is_equal = _assert_nan_coords_same(x, y, tolerance=tolerance)
    if not np.all(is_equal):
        raise AssertionError("One of the geometries are not equal")
