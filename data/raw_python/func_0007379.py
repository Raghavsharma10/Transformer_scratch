def number_to_bytes(n, endian='big'):
    """
    Convert an integer to a corresponding string of bytes..

    :param n:
        Integer to convert.

    :param endian:
        Byte order to convert into ('big' or 'little' endian-ness, default
        'big')

    Assumes bytes are 8 bits.

    This is a special-case version of number_to_string with a full base-256
    ASCII alphabet. It is the reverse of ``bytes_to_number(b)``.

    Examples::

        >>> r(number_to_bytes(42))
        b'*'
        >>> r(number_to_bytes(255))
        b'\\xff'
        >>> r(number_to_bytes(256))
        b'\\x01\\x00'
        >>> r(number_to_bytes(256, endian='little'))
        b'\\x00\\x01'
    """
    res = []
    while n:
        n, ch = divmod(n, 256)
        if PY3:
            res.append(ch)
        else:
            res.append(chr(ch))

    if endian == 'big':
        res.reverse()

    if PY3:
        return bytes(res)
    else:
        return ''.join(res)