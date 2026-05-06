def _unbytes(bytestr):
    """
    Returns a bytestring from the human-friendly string returned by `_bytes`.

    >>> _unbytes('123456')
    '\x12\x34\x56'
    """
    return ''.join(chr(int(bytestr[k:k + 2], 16))
                   for k in range(0, len(bytestr), 2))