def fitString(s, maxCol=79, newlineReplacement=None):
    r"""Truncate `s` if necessary to fit into a line of width `maxCol`
    (default: 79), also replacing newlines with `newlineReplacement` (default
    `None`: in which case everything after the first newline is simply
    discarded).

    Examples:

    >>> fitString('12345', maxCol=5)
    '12345'
    >>> fitString('123456', maxCol=5)
    '12...'
    >>> fitString('a line\na second line')
    'a line'
    >>> fitString('a line\na second line', newlineReplacement='\\n')
    'a line\\na second line'
    """
    assert isString(s)
    if '\n' in s:
        if newlineReplacement is None:
            s = s[:s.index('\n')]
        else:
            s = s.replace("\n", newlineReplacement)
    if maxCol is not None and len(s) > maxCol:
        s = "%s..." % s[:maxCol-3]
    return s