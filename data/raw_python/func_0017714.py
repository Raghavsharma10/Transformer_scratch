def validate_cidr(s):
    """Validate a CIDR notation ip address.

    The string is considered a valid CIDR address if it consists of a valid
    IPv6 address in hextet format followed by a forward slash (/) and a bit
    mask length (0-128).


    >>> validate_cidr('::/128')
    True
    >>> validate_cidr('::/0')
    True
    >>> validate_cidr('fc00::/7')
    True
    >>> validate_cidr('::ffff:0:0/96')
    True
    >>> validate_cidr('::')
    False
    >>> validate_cidr('::/129')
    False
    >>> validate_cidr(None) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected string or buffer


    :param s: String to validate as a CIDR notation ip address.
    :type s: str
    :returns: ``True`` if a valid CIDR address, ``False`` otherwise.
    :raises: TypeError
    """
    if _CIDR_RE.match(s):
        ip, mask = s.split('/')
        if validate_ip(ip):
            if int(mask) > 128:
                return False
        else:
            return False
        return True
    return False