from contextlib import contextmanager
import numpy as np

@contextmanager
def not_writable(arr):
    if isinstance(arr, np.ndarray):
        old = arr.flags.writeable
        arr.flags.writeable = False
        try:
            yield
        finally:
            arr.flags.writeable = old
    else:
        yield
