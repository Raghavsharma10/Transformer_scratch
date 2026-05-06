def subnet2block(subnet):
    """Convert a dotted-quad ip address including a netmask into a tuple
    containing the network block start and end addresses.


    >>> subnet2block('127.0.0.1/255.255.255.255')
    ('127.0.0.1', '127.0.0.1')
    >>> subnet2block('127/255')
    ('127.0.0.0', '127.255.255.255')
    >>> subnet2block('127.0.1/255.255')
    ('127.0.0.0', '127.0.255.255')
    >>> subnet2block('127.1/255.255.255.0')
    ('127.1.0.0', '127.1.0.255')
    >>> subnet2block('127.0.0.3/255.255.255.248')
    ('127.0.0.0', '127.0.0.7')
    >>> subnet2block('127/0')
    ('0.0.0.0', '255.255.255.255')


    :param subnet: dotted-quad ip address with netmask
        (eg. '127.0.0.1/255.0.0.0').
    :type subnet: str
    :returns: Tuple of block (start, end) or ``None`` if invalid.
    :raises: TypeError
    """
    if not validate_subnet(subnet):
        return None

    ip, netmask = subnet.split('/')
    prefix = netmask2prefix(netmask)

    # convert dotted-quad ip to base network number
    network = ip2network(ip)

    return _block_from_ip_and_prefix(network, prefix)