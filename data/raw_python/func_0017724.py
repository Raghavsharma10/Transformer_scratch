def validate_ip(s):
    """Validate a dotted-quad ip address.

    The string is considered a valid dotted-quad address if it consists of
    one to four octets (0-255) seperated by periods (.).


    >>> validate_ip('127.0.0.1')
    True
    >>> validate_ip('127.0')
    True
    >>> validate_ip('127.0.0.256')
    False
    >>> validate_ip(LOCALHOST)
    True
    >>> validate_ip(None) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected string or buffer


    :param s: String to validate as a dotted-quad ip address.
    :type s: str
    :returns: ``True`` if a valid dotted-quad ip address, ``False`` otherwise.
    :raises: TypeError
    """
    if _DOTTED_QUAD_RE.match(s):
        quads = s.split('.')
        for q in quads:
            if int(q) > 255:
                return False
        return True
    return False