def stringify_address(addr, encoding='utf-8'):
    """
    Given an email address *addr*, try to encode
    it with ASCII. If it's not possible, encode
    the *local-part* with the *encoding* and the
    *domain* with IDNA.

    The result is a unicode string with the domain
    encoded as idna.
    """
    if isinstance(addr, bytes_type):
        return addr
    try:
        addr = addr.encode('ascii')
    except UnicodeEncodeError:
        if '@' in addr:
            localpart, domain = addr.split('@', 1)
            addr = b'@'.join([
                localpart.encode(encoding),
                domain.encode('idna'),
            ])
        else:
            addr = addr.encode(encoding)
    return addr.decode('utf-8')