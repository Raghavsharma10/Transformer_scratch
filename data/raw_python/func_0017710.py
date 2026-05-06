def ip2long(ip):
    """Convert a hexidecimal IPv6 address to a network byte order 128-bit
    integer.


    >>> ip2long('::') == 0
    True
    >>> ip2long('::1') == 1
    True
    >>> expect = 0x20010db885a3000000008a2e03707334
    >>> ip2long('2001:db8:85a3::8a2e:370:7334') == expect
    True
    >>> ip2long('2001:db8:85a3:0:0:8a2e:370:7334') == expect
    True
    >>> ip2long('2001:0db8:85a3:0000:0000:8a2e:0370:7334') == expect
    True
    >>> expect = 0x20010db8000000000001000000000001
    >>> ip2long('2001:db8::1:0:0:1') == expect
    True
    >>> expect = 281473902969472
    >>> ip2long('::ffff:192.0.2.128') == expect
    True
    >>> expect = 0xffffffffffffffffffffffffffffffff
    >>> ip2long('ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff') == expect
    True
    >>> ip2long('ff::ff::ff') == None
    True
    >>> expect = 21932261930451111902915077091070067066
    >>> ip2long('1080:0:0:0:8:800:200C:417A') == expect
    True


    :param ip: Hexidecimal IPv6 address
    :type ip: str
    :returns: Network byte order 128-bit integer or ``None`` if ip is invalid.
    """
    if not validate_ip(ip):
        return None

    if '.' in ip:
        # convert IPv4 suffix to hex
        chunks = ip.split(':')
        v4_int = ipv4.ip2long(chunks.pop())
        if v4_int is None:
            return None
        chunks.append('%x' % ((v4_int >> 16) & 0xffff))
        chunks.append('%x' % (v4_int & 0xffff))
        ip = ':'.join(chunks)

    halves = ip.split('::')
    hextets = halves[0].split(':')
    if len(halves) == 2:
        h2 = halves[1].split(':')
        for z in range(8 - (len(hextets) + len(h2))):
            hextets.append('0')
        for h in h2:
            hextets.append(h)
    # end if

    lngip = 0
    for h in hextets:
        if '' == h:
            h = '0'
        lngip = (lngip << 16) | int(h, 16)
    return lngip