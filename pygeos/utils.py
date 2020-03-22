from contextlib import contextmanager
from itertools import chain
from functools import wraps
import numpy as np


def multithreading_enabled(func):
    """Prepare multithreading by setting the writable flags of object type
    ndarrays to False.

    NB: multithreading also requires the GIL to be released, which is done in
    the C extension (ufuncs.c)."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        array_args = [
            arg for arg in chain(args, kwargs.values())
            if isinstance(arg, np.ndarray) and arg.dtype == object
        ]
        old_flags = [arr.flags.writeable for arr in array_args]
        try:
            for arr in array_args:
                arr.flags.writeable = False
            return func(*args, **kwargs)
        finally:
            for arr, old_flag in zip(array_args, old_flags):
                arr.flags.writeable = old_flag
    return wrapped
