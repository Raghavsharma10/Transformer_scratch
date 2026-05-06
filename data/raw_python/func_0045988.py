def get_regex(regex):
    """
    Ensure we have a compiled regular expression object.

        >>> import re
        >>> get_regex('string') # doctest: +ELLIPSIS
        <_sre.SRE_Pattern object at 0x...>
        >>> pattern = re.compile(r'string')
        >>> get_regex(pattern) is pattern
        True
        >>> get_regex(3) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: Invalid regex type: 3
    """
    if isinstance(regex, basestring):
        return re.compile(regex)
    elif not isinstance(regex, re._pattern_type):
        raise TypeError("Invalid regex type: %r" % (regex,))
    return regex