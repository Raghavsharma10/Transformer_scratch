def ip2network(ip):
    """Convert a dotted-quad ip to base network number.

    This differs from :func:`ip2long` in that partial addresses as treated as
    all network instead of network plus host (eg. '127.1' expands to
    '127.1.0.0')

    :param ip: dotted-quad ip address (eg. ‘127.0.0.1’).
    :type ip: str
    :returns: Network byte order 32-bit integer or `None` if ip is invalid.
    """
    if not validate_ip(ip):
        return None
    quads = ip.split('.')
    netw = 0
    for i in range(4):
        netw = (netw << 8) | int(len(quads) > i and quads[i] or 0)
    return netw