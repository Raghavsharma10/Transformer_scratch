def is_iterable(maybe_iter, unless=(string_types, dict)):
    """ Return whether ``maybe_iter`` is an iterable, unless it's an instance of one
    of the base class, or tuple of base classes, given in ``unless``.

    Example::

        >>> is_iterable('foo')
        False
        >>> is_iterable(['foo'])
        True
        >>> is_iterable(['foo'], unless=list)
        False
        >>> is_iterable(xrange(5))
        True
    """
    try:
        iter(maybe_iter)
    except TypeError:
        return False
    return not isinstance(maybe_iter, unless)