def _address2long(address):
    """
    Convert an address string to a long.
    """
    parsed = ipv4.ip2long(address)
    if parsed is None:
        parsed = ipv6.ip2long(address)
    return parsed