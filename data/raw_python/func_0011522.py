def bytes_to_bits(bytes_):
    """Convert bytes to a list of bits
    """
    res = []
    for x in bytes_:
        if not isinstance(x, int):
            x = ord(x)
        res += byte_to_bits(x)
    return res