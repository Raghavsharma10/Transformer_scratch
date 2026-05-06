def deprecated(message, exception=PendingDeprecationWarning):
    """Throw a warning when a function/method will be soon deprecated

    Supports passing a ``message`` and an ``exception`` class
    (uses ``PendingDeprecationWarning`` by default). This is useful if you
    want to alternatively pass a ``DeprecationWarning`` exception for already
    deprecated functions/methods.

    Example::

        >>> import warnings
        >>> from functools import wraps
        >>> message = "this function will be deprecated in the near future"
        >>> @deprecated(message)
        ... def foo(n):
        ...     return n+n
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     foo(4)
        ...     assert len(w) == 1
        ...     assert issubclass(w[-1].category, PendingDeprecationWarning)
        ...     assert message == str(w[-1].message)
        ...     assert foo.__name__ == 'foo'
        8
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, exception, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator