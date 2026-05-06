def validate_subnet(s):
    """Validate a dotted-quad ip address including a netmask.

    The string is considered a valid dotted-quad address with netmask if it
    consists of one to four octets (0-255) seperated by periods (.) followed
    by a forward slash (/) and a subnet bitmask which is expressed in
    dotted-quad format.


    >>> validate_subnet('127.0.0.1/255.255.255.255')
    True
    >>> validate_subnet('127.0/255.0.0.0')
    True
    >>> validate_subnet('127.0/255')
    True
    >>> validate_subnet('127.0.0.256/255.255.255.255')
    False
    >>> validate_subnet('127.0.0.1/255.255.255.256')
    False
    >>> validate_subnet('127.0.0.0')
    False
    >>> validate_subnet(None)
    Traceback (most recent call last):
        ...
    TypeError: expected string or unicode


    :param s: String to validate as a dotted-quad ip address with netmask.
    :type s: str
    :returns: ``True`` if a valid dotted-quad ip address with netmask,
        ``False`` otherwise.
    :raises: TypeError
    """
    if isinstance(s, basestring):
        if '/' in s:
            start, mask = s.split('/', 2)
            return validate_ip(start) and validate_netmask(mask)
        else:
            return False
    raise TypeError("expected string or unicode")