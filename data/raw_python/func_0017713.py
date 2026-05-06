def rfc19242long(s):
    """Convert an RFC 1924 IPv6 address to a network byte order 128-bit
    integer.


    >>> expect = 0
    >>> rfc19242long('00000000000000000000') == expect
    True
    >>> expect = 21932261930451111902915077091070067066
    >>> rfc19242long('4)+k&C#VzJ4br>0wv%Yp') == expect
    True
    >>> rfc19242long('pizza') == None
    True
    >>> rfc19242long('~~~~~~~~~~~~~~~~~~~~') == None
    True
    >>> rfc19242long('=r54lj&NUUO~Hi%c2ym0') == MAX_IP
    True


    :param ip: RFC 1924  IPv6 address
    :type ip: str
    :returns: Network byte order 128-bit integer or ``None`` if ip is invalid.
    """
    global _RFC1924_REV
    if not _RFC1924_RE.match(s):
        return None
    if _RFC1924_REV is None:
        _RFC1924_REV = {v: k for k, v in enumerate(_RFC1924_ALPHABET)}
    x = 0
    for c in s:
        x = x * 85 + _RFC1924_REV[c]
    if x > MAX_IP:
        return None
    return x