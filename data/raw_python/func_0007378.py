def bytes_to_number(b, endian='big'):
    """
    Convert a string to an integer.

    :param b:
        String or bytearray to convert.

    :param endian:
        Byte order to convert into ('big' or 'little' endian-ness, default
        'big')

    Assumes bytes are 8 bits.

    This is a special-case version of string_to_number with a full base-256
    ASCII alphabet. It is the reverse of ``number_to_bytes(n)``.

    Examples::

        >>> bytes_to_number(b'*')
        42
        >>> bytes_to_number(b'\\xff')
        255
        >>> bytes_to_number(b'\\x01\\x00')
        256
        >>> bytes_to_number(b'\\x00\\x01', endian='little')
        256
    """
    if endian == 'big':
        b = reversed(b)

    n = 0
    for i, ch in enumerate(bytearray(b)):
        n ^= ch << i * 8

    return n