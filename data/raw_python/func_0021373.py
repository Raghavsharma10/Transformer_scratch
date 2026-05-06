def setattr_context(obj, **kwargs):
    """
    Context manager to temporarily change the values of object attributes
    while executing a function.

    Example
    -------
    >>> class Foo: pass
    >>> f = Foo(); f.attr = 'hello'
    >>> with setattr_context(f, attr='goodbye'):
    ...     print(f.attr)
    goodbye
    >>> print(f.attr)
    hello
    """
    old_kwargs = dict([(key, getattr(obj, key)) for key in kwargs])
    [setattr(obj, key, val) for key, val in kwargs.items()]
    try:
        yield
    finally:
        [setattr(obj, key, val) for key, val in old_kwargs.items()]