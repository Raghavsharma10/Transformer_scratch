def cidr2block(cidr):
    """Convert a CIDR notation ip address into a tuple containing the network
    block start and end addresses.


    >>> cidr2block('2001:db8::/48')
    ('2001:db8::', '2001:db8:0:ffff:ffff:ffff:ffff:ffff')
    >>> cidr2block('::/0')
    ('::', 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff')


    :param cidr: CIDR notation ip address (eg. '127.0.0.1/8').
    :type cidr: str
    :returns: Tuple of block (start, end) or ``None`` if invalid.
    :raises: TypeError
    """
    if not validate_cidr(cidr):
        return None

    ip, prefix = cidr.split('/')
    prefix = int(prefix)
    ip = ip2long(ip)

    # keep left most prefix bits of ip
    shift = 128 - prefix
    block_start = ip >> shift << shift

    # expand right most 128 - prefix bits to 1
    mask = (1 << shift) - 1
    block_end = block_start | mask
    return (long2ip(block_start), long2ip(block_end))