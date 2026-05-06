def hex_from(val):
    """Returns hex string representation for a given value.

    :param bytes|str|unicode|int|long val:
    :rtype: bytes|str
    """
    if isinstance(val, integer_types):
        hex_str = '%x' % val
        if len(hex_str) % 2:
            hex_str = '0' + hex_str
        return hex_str

    return hexlify(val)