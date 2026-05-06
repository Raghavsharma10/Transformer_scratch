def ip2long(ip):
    """Convert a dotted-quad ip address to a network byte order 32-bit
    integer.


    >>> ip2long('127.0.0.1')
    2130706433
    >>> ip2long('127.1')
    2130706433
    >>> ip2long('127')
    2130706432
    >>> ip2long('127.0.0.256') is None
    True


    :param ip: Dotted-quad ip address (eg. '127.0.0.1').
    :type ip: str
    :returns: Network byte order 32-bit integer or ``None`` if ip is invalid.
    """
    if not validate_ip(ip):
        return None
    quads = ip.split('.')
    if len(quads) == 1:
        # only a network quad
        quads = quads + [0, 0, 0]
    elif len(quads) < 4:
        # partial form, last supplied quad is host address, rest is network
        host = quads[-1:]
        quads = quads[:-1] + [0, ] * (4 - len(quads)) + host

    lngip = 0
    for q in quads:
        lngip = (lngip << 8) | int(q)
    return lngip