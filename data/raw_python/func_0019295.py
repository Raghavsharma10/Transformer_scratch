def assignrepr_values(values, prefix, width=None, _fakeend=0):
    """Return a prefixed, wrapped and properly aligned string representation
    of the given values using function |repr|.

    >>> from hydpy.core.objecttools import assignrepr_values
    >>> print(assignrepr_values(range(1, 13), 'test(', 20) + ')')
    test(1, 2, 3, 4, 5,
         6, 7, 8, 9, 10,
         11, 12)

    If no width is given, no wrapping is performed:

    >>> print(assignrepr_values(range(1, 13), 'test(') + ')')
    test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


    To circumvent defining too long string representations, make use of the
    ellipsis option:

    >>> from hydpy import pub
    >>> with pub.options.ellipsis(1):
    ...     print(assignrepr_values(range(1, 13), 'test(', 20) + ')')
    test(1, ...,12)

    >>> with pub.options.ellipsis(5):
    ...     print(assignrepr_values(range(1, 13), 'test(', 20) + ')')
    test(1, 2, 3, 4, 5,
         ...,8, 9, 10,
         11, 12)

    >>> with pub.options.ellipsis(6):
    ...     print(assignrepr_values(range(1, 13), 'test(', 20) + ')')
    test(1, 2, 3, 4, 5,
         6, 7, 8, 9, 10,
         11, 12)
    """
    ellipsis_ = hydpy.pub.options.ellipsis
    if (ellipsis_ > 0) and (len(values) > 2*ellipsis_):
        string = (repr_values(values[:ellipsis_]) +
                  ', ...,' +
                  repr_values(values[-ellipsis_:]))
    else:
        string = repr_values(values)
    blanks = ' '*len(prefix)
    if width is None:
        wrapped = [string]
        _fakeend = 0
    else:
        width -= len(prefix)
        wrapped = textwrap.wrap(string+'_'*_fakeend, width)
    if not wrapped:
        wrapped = ['']
    lines = []
    for (idx, line) in enumerate(wrapped):
        if idx == 0:
            lines.append('%s%s' % (prefix, line))
        else:
            lines.append('%s%s' % (blanks, line))
    string = '\n'.join(lines)
    return string[:len(string)-_fakeend]