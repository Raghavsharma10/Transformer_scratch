def parseAddress(address):
    """
    Parse the given RFC 2821 email address into a structured object.

    @type address: C{str}
    @param address: The address to parse.

    @rtype: L{Address}

    @raise xmantissa.error.ArgumentError: The given string was not a valid RFC
    2821 address.
    """
    parts = []
    parser = _AddressParser()
    end = parser(parts, address)
    if end != len(address):
        raise InvalidTrailingBytes()
    return parts[0]