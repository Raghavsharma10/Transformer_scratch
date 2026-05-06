def validate_ip(s):
    """Validate a hexidecimal IPv6 ip address.


    >>> validate_ip('::')
    True
    >>> validate_ip('::1')
    True
    >>> validate_ip('2001:db8:85a3::8a2e:370:7334')
    True
    >>> validate_ip('2001:db8:85a3:0:0:8a2e:370:7334')
    True
    >>> validate_ip('2001:0db8:85a3:0000:0000:8a2e:0370:7334')
    True
    >>> validate_ip('2001:db8::1:0:0:1')
    True
    >>> validate_ip('ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff')
    True
    >>> validate_ip('::ffff:192.0.2.128')
    True
    >>> validate_ip('::ff::ff')
    False
    >>> validate_ip('::fffff')
    False
    >>> validate_ip('::ffff:192.0.2.300')
    False
    >>> validate_ip(None) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected string or buffer
    >>> validate_ip('1080:0:0:0:8:800:200c:417a')
    True


    :param s: String to validate as a hexidecimal IPv6 ip address.
    :type s: str
    :returns: ``True`` if a valid hexidecimal IPv6 ip address,
              ``False`` otherwise.
    :raises: TypeError
    """
    if _HEX_RE.match(s):
        return len(s.split('::')) <= 2
    if _DOTTED_QUAD_RE.match(s):
        halves = s.split('::')
        if len(halves) > 2:
            return False
        hextets = s.split(':')
        quads = hextets[-1].split('.')
        for q in quads:
            if int(q) > 255:
                return False
        return True
    return False