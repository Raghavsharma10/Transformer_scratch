def wrap_exception(exc, new_exc):
    """Catches exceptions `exc` and raises `new_exc(exc)` instead.

    E.g.::

        >>> class MyValueError(Exception):
        ... '''Custom ValueError.'''
        ... @wrap_exception(exc=ValueError, new_exc=MyValueError)
        ... def test():
        ...    raise ValueError()

    """
    def make_wrapper(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            try:
                return fn(*args, **kw)
            except exc as e:
                future.utils.raise_with_traceback(new_exc(e))
        return wrapper
    return make_wrapper