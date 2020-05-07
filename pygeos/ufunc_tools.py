UFUNC_ATTRIBUTES = {
    'nin': None,
    'nout': None,
    'nargs': None,
    'ntypes': None,
    'types': None,
    'identity': None,
    'signature': None,
}

UFUNC_METHODS = {
    'reduce': None,
    'accumulate': None,
    'reduceat': None,
    'outer': None,
    'at': None,
}


def ufunc_attributes_wrapper(destination, attribtues=None):
    """Wraps func to expose underlaying ufunc attributes.
    
    Parameters
    ----------
    destination : original ufunc
        The original ufunc that `func` wraps. If `attributes` is None,
        all attributes of `destination` will be exposed.
    attribtues : dict, default None
        Dictionary mapping desired attributes to their values. If None,
        the attributes of `destination` will be used.
    """
    if attribtues is None:
        attribtues = UFUNC_ATTRIBUTES
    def decorator(func):
        for attribute, val in attribtues.items():
            setattr(func, attribute, val or getattr(destination, attribute))
        return func
    return decorator


def ufunc_methods_wrapper(destination, methods=None):
    """Wraps func to expose underlaying ufunc methods.
    
    Parameters
    ----------
    destination : original ufunc
        The original ufunc that `func` wraps. If `methods` is None,
        all methods of `destination` will be exposed.
    methods : dict, default None
        Dictionary mapping desired methods to their values. If None,
        the methods of `destination` will be used.
    """
    if methods is None:
        methods = UFUNC_METHODS
    def decorator(func):
        for attribute, val in methods.items():
            setattr(func, attribute, val or getattr(destination, attribute))
        return func
    return decorator


class UfuncWrapper:

    def __init__(self, ufunc):
        self.ufunc = ufunc

    def __call__(self, *args, **kwargs):
        return self.ufunc(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return self.ufunc.reduce(*args, **kwargs)