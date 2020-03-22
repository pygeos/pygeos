from . import lib
from functools import wraps


class UnsupportedGEOSOperation(ImportError):
    pass


class requires_geos:
    def __init__(self, version):
        if version.count(".") == 1:
            version += ".0"
        self.version = version

    def __call__(self, func):
        if lib.geos_version < tuple(int(x) for x in self.version.split(".")):
            msg = "'{}' requires at least GEOS {}".format(func.__name__, self.version)

            @wraps(func)
            def wrapped(*args, **kwargs):
                raise UnsupportedGEOSOperation(msg)

            wrapped.__doc__ = msg
            return wrapped
        else:
            return func
