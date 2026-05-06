def _block_from_ip_and_prefix(ip, prefix):
    """Create a tuple of (start, end) dotted-quad addresses from the given
    ip address and prefix length.

    :param ip: Ip address in block
    :type ip: long
    :param prefix: Prefix size for block
    :type prefix: int
    :returns: Tuple of block (start, end)
    """
    # keep left most prefix bits of ip
    shift = 32 - prefix
    block_start = ip >> shift << shift

    # expand right most 32 - prefix bits to 1
    mask = (1 << shift) - 1
    block_end = block_start | mask
    return (long2ip(block_start), long2ip(block_end))