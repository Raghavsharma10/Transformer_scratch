def long2rfc1924(l):
    """Convert a network byte order 128-bit integer to an rfc1924 IPv6
    address.


    >>> long2rfc1924(ip2long('1080::8:800:200C:417A'))
    '4)+k&C#VzJ4br>0wv%Yp'
    >>> long2rfc1924(ip2long('::'))
    '00000000000000000000'
    >>> long2rfc1924(MAX_IP)
    '=r54lj&NUUO~Hi%c2ym0'


    :param l: Network byte order 128-bit integer.
    :type l: int
    :returns: RFC 1924 IPv6 address
    :raises: TypeError
    """
    if MAX_IP < l or l < MIN_IP:
        raise TypeError(
            "expected int between %d and %d inclusive" % (MIN_IP, MAX_IP))
    o = []
    r = l
    while r > 85:
        o.append(_RFC1924_ALPHABET[r % 85])
        r = r // 85
    o.append(_RFC1924_ALPHABET[r])
    return ''.join(reversed(o)).zfill(20)