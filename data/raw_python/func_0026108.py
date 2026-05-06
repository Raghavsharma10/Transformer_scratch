def tobytes(s, encoding='ascii'):
    """ Convert string s to the 'bytes' type, in all Pythons, even
    back before Python 2.6.  What 'str' means varies by PY3K or not.
    In Pythons before 3.0, this is technically the same as the str type
    in terms of the character data in memory. """
    # NOTE: after we abandon 2.5, we might simply instead use "bytes(s)"
    # NOTE: after we abandon all 2.*, del this and prepend byte strings with 'b'
    if PY3K:
        if isinstance(s, bytes):
            return s
        else:
            return s.encode(encoding)
    else:
        # for py2.6 on (before 3.0), bytes is same as str;  2.5 has no bytes
        # but handle if unicode is passed
        if isinstance(s, unicode):
            return s.encode(encoding)
        else:
            return s