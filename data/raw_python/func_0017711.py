def long2ip(l, rfc1924=False):
    """Convert a network byte order 128-bit integer to a canonical IPv6
    address.


    >>> long2ip(2130706433)
    '::7f00:1'
    >>> long2ip(42540766411282592856904266426630537217)
    '2001:db8::1:0:0:1'
    >>> long2ip(MIN_IP)
    '::'
    >>> long2ip(MAX_IP)
    'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'
    >>> long2ip(None) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for >>: 'NoneType' and 'int'
    >>> long2ip(-1) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected int between 0 and <really big int> inclusive
    >>> long2ip(MAX_IP + 1) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: expected int between 0 and <really big int> inclusive
    >>> long2ip(ip2long('1080::8:800:200C:417A'), rfc1924=True)
    '4)+k&C#VzJ4br>0wv%Yp'
    >>> long2ip(ip2long('::'), rfc1924=True)
    '00000000000000000000'


    :param l: Network byte order 128-bit integer.
    :type l: int
    :param rfc1924: Encode in RFC 1924 notation (base 85)
    :type rfc1924: bool
    :returns: Canonical IPv6 address (eg. '::1').
    :raises: TypeError
    """
    if MAX_IP < l or l < MIN_IP:
        raise TypeError(
            "expected int between %d and %d inclusive" % (MIN_IP, MAX_IP))

    if rfc1924:
        return long2rfc1924(l)

    # format as one big hex value
    hex_str = '%032x' % l
    # split into double octet chunks without padding zeros
    hextets = ['%x' % int(hex_str[x:x + 4], 16) for x in range(0, 32, 4)]

    # find and remove left most longest run of zeros
    dc_start, dc_len = (-1, 0)
    run_start, run_len = (-1, 0)
    for idx, hextet in enumerate(hextets):
        if '0' == hextet:
            run_len += 1
            if -1 == run_start:
                run_start = idx
            if run_len > dc_len:
                dc_len, dc_start = (run_len, run_start)
        else:
            run_len, run_start = (0, -1)
    # end for
    if dc_len > 1:
        dc_end = dc_start + dc_len
        if dc_end == len(hextets):
            hextets += ['']
        hextets[dc_start:dc_end] = ['']
        if dc_start == 0:
            hextets = [''] + hextets
    # end if

    return ':'.join(hextets)