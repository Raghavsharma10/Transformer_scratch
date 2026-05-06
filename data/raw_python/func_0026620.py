def AlternatesGroup(expressions, final_function, name=""):
    """ Group expressions using the OR character ``|``
    >>> from collections import namedtuple
    >>> expr = namedtuple('expr', 'regex group_lengths run')('(1)', [1], None)
    >>> grouping = AlternatesGroup([expr, expr], lambda f: None, 'yeah')
    >>> grouping.regex  # doctest: +IGNORE_UNICODE
    '(?:(1))|(?:(1))'
    >>> grouping.group_lengths
    [1, 1]
    """
    inbetweens = ["|"] * (len(expressions) + 1)
    inbetweens[0] = ""
    inbetweens[-1] = ""
    return Group(expressions, final_function, inbetweens, name)