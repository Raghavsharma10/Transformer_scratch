def mask(key, original):
    """
    Mask an octet string using the given masking key.
    The following masking algorithm is used, as defined in RFC 6455:

    for each octet:
        j = i MOD 4
        transformed-octet-i = original-octet-i XOR masking-key-octet-j
    """
    if len(key) != 4:
        raise ValueError('invalid masking key "%s"' % key)

    key = map(ord, key)
    masked = bytearray(original)

    for i in xrange(len(masked)):
        masked[i] ^= key[i % 4]

    return masked