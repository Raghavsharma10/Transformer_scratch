def long2ip(l):
    """Convert a network byte order 32-bit integer to a dotted quad ip
    address.


    >>> long2ip(2130706433)
    '127.0.0.1'
    >>> long2ip(MIN_IP)
    '0.0.0.0'
    >>> long2ip(MAX_IP)
    '255.255.255.255'
    >>> long2ip(None) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for >>: 'NoneType' and 'int'
    >>> long2ip(-1) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected int between 0 and 4294967295 inclusive
    >>> long2ip(374297346592387463875) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected int between 0 and 4294967295 inclusive
    >>> long2ip(MAX_IP + 1) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected int between 0 and 4294967295 inclusive


    :param l: Network byte order 32-bit integer.
    :type l: int
    :returns: Dotted-quad ip address (eg. '127.0.0.1').
    :raises: TypeError
    """
    if MAX_IP < l or l < MIN_IP:
        raise TypeError(
            "expected int between %d and %d inclusive" % (MIN_IP, MAX_IP))
    return '%d.%d.%d.%d' % (
        l >> 24 & 255, l >> 16 & 255, l >> 8 & 255, l & 255)