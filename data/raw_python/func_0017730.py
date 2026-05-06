def cidr2block(cidr):
    """Convert a CIDR notation ip address into a tuple containing the network
    block start and end addresses.


    >>> cidr2block('127.0.0.1/32')
    ('127.0.0.1', '127.0.0.1')
    >>> cidr2block('127/8')
    ('127.0.0.0', '127.255.255.255')
    >>> cidr2block('127.0.1/16')
    ('127.0.0.0', '127.0.255.255')
    >>> cidr2block('127.1/24')
    ('127.1.0.0', '127.1.0.255')
    >>> cidr2block('127.0.0.3/29')
    ('127.0.0.0', '127.0.0.7')
    >>> cidr2block('127/0')
    ('0.0.0.0', '255.255.255.255')


    :param cidr: CIDR notation ip address (eg. '127.0.0.1/8').
    :type cidr: str
    :returns: Tuple of block (start, end) or ``None`` if invalid.
    :raises: TypeError
    """
    if not validate_cidr(cidr):
        return None

    ip, prefix = cidr.split('/')
    prefix = int(prefix)

    # convert dotted-quad ip to base network number
    network = ip2network(ip)

    return _block_from_ip_and_prefix(network, prefix)