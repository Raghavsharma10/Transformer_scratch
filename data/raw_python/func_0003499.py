def func_call_as_str(name, *args, **kwds):
    """
    Return arguments and keyword arguments as formatted string

    >>> func_call_as_str('f', 1, 2, a=1)
    'f(1, 2, a=1)'

    """
    return '{0}({1})'.format(
        name,
        ', '.join(itertools.chain(
            map('{0!r}'.format, args),
            map('{0[0]!s}={0[1]!r}'.format, sorted(kwds.items())))))