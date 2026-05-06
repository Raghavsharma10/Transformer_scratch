def iterate(maybe_iter, unless=(string_types, dict)):
    """ Always return an iterable.

    Returns ``maybe_iter`` if it is an iterable, otherwise it returns a single
    element iterable containing ``maybe_iter``. By default, strings and dicts
    are treated as non-iterable. This can be overridden by passing in a type
    or tuple of types for ``unless``.

    :param maybe_iter:
        A value to return as an iterable.

    :param unless:
        A type or tuple of types (same as ``isinstance``) to be treated as
        non-iterable.

    Example::

        >>> iterate('foo')
        ['foo']
        >>> iterate(['foo'])
        ['foo']
        >>> iterate(['foo'], unless=list)
        [['foo']]
        >>> list(iterate(xrange(5)))
        [0, 1, 2, 3, 4]
    """
    if is_iterable(maybe_iter, unless=unless):
        return maybe_iter
    return [maybe_iter]