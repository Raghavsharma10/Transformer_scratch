def _default_hashfunc(content, hashbits):
    """
    Default hash function is variable-length version of Python's builtin hash.

    :param content: data that needs to hash.
    :return: return a decimal number.
    """
    if content == "":
        return 0

    x = ord(content[0]) << 7
    m = 1000003
    mask = 2 ** hashbits - 1
    for c in content:
        x = ((x * m) ^ ord(c)) & mask
    x ^= len(content)
    if x == -1:
        x = -2
    return x