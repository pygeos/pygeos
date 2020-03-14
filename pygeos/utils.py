from contextlib import contextmanager
import numpy as np

@contextmanager
def not_writable(arr):
    if not isinstance(arr, np.ndarray):
        yield
    old = arr.flags.writeable
    arr.flags.writeable = False
    try:
        yield
    finally:
        arr.flags.writeable = old
