def assert_hashable(*args, **kw):
    """ Verify that each argument is hashable.

    Passes silently if successful. Raises descriptive TypeError otherwise.

    Example::

        >>> assert_hashable(1, 'foo', bar='baz')
        >>> assert_hashable(1, [], baz='baz')
        Traceback (most recent call last):
          ...
        TypeError: Argument in position 1 is not hashable: []
        >>> assert_hashable(1, 'foo', bar=[])
        Traceback (most recent call last):
          ...
        TypeError: Keyword argument 'bar' is not hashable: []
    """
    try:
        for i, arg in enumerate(args):
            hash(arg)
    except TypeError:
        raise TypeError('Argument in position %d is not hashable: %r' % (i, arg))
    try:
        for key, val in iterate_items(kw):
            hash(val)
    except TypeError:
        raise TypeError('Keyword argument %r is not hashable: %r' % (key, val))