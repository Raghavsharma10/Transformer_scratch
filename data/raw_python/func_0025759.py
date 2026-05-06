def numToDottedQuad(num):
    """
    Convert long int to dotted quad string

    >>> numToDottedQuad(long(-1))
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: -1
    >>> numToDottedQuad(long(1))
    '0.0.0.1'
    >>> numToDottedQuad(long(16777218))
    '1.0.0.2'
    >>> numToDottedQuad(long(16908291))
    '1.2.0.3'
    >>> numToDottedQuad(long(16909060))
    '1.2.3.4'
    >>> numToDottedQuad(long(4294967295))
    '255.255.255.255'
    >>> numToDottedQuad(long(4294967296))
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: 4294967296
    """

    # import here to avoid it when ip_addr values are not used
    import socket, struct

    # no need to intercept here, 4294967295 is fine
    if num > 4294967295 or num < 0:
        raise ValueError('Not a good numeric IP: %s' % num)
    try:
        return socket.inet_ntoa(
            struct.pack('!L', long(num)))
    except (socket.error, struct.error, OverflowError):
        raise ValueError('Not a good numeric IP: %s' % num)