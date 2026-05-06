def validate_netmask(s):
    """Validate that a dotted-quad ip address is a valid netmask.


    >>> validate_netmask('0.0.0.0')
    True
    >>> validate_netmask('128.0.0.0')
    True
    >>> validate_netmask('255.0.0.0')
    True
    >>> validate_netmask('255.255.255.255')
    True
    >>> validate_netmask(BROADCAST)
    True
    >>> validate_netmask('128.0.0.1')
    False
    >>> validate_netmask('1.255.255.0')
    False
    >>> validate_netmask('0.255.255.0')
    False


    :param s: String to validate as a dotted-quad notation netmask.
    :type s: str
    :returns: ``True`` if a valid netmask, ``False`` otherwise.
    :raises: TypeError
    """
    if validate_ip(s):
        # Convert to binary string, strip '0b' prefix, 0 pad to 32 bits
        mask = bin(ip2network(s))[2:].zfill(32)
        # all left most bits must be 1, all right most must be 0
        seen0 = False
        for c in mask:
            if '1' == c:
                if seen0:
                    return False
            else:
                seen0 = True
        return True
    else:
        return False