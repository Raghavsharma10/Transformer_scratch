def elfhash(s):
    """
    :param string: bytes

    >>> import base64
    >>> s = base64.b64encode(b'hello world')
    >>> elfhash(s)
    224648685
    """
    hash = 0
    x = 0
    for c in s:
        hash = (hash << 4) + c
        x = hash & 0xF0000000
        if x:
            hash ^= (x >> 24)
            hash &= ~x
    return (hash & 0x7FFFFFFF)