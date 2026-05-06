def deprecated(function, instead):
    """Mark a function deprecated so calling it issues a warning"""

    # skip for classes, breaks doc generation
    if not isinstance(function, types.FunctionType):
        return function

    @wraps(function)
    def wrap(*args, **kwargs):
        warnings.warn("Deprecated, use %s instead" % instead,
                      PyGIDeprecationWarning)
        return function(*args, **kwargs)

    return wrap