def tostr(s, encoding='ascii'):
    """ Convert string-like-thing s to the 'str' type, in all Pythons, even
    back before Python 2.6.  What 'str' means varies by PY3K or not.
    In Pythons before 3.0, str and bytes are the same type.
    In Python 3+, this may require a decoding step. """
    if PY3K:
        if isinstance(s, str): # str == unicode in PY3K
            return s
        else: # s is type bytes
            return s.decode(encoding)
    else:
        # for py2.6 on (before 3.0), bytes is same as str;  2.5 has no bytes
        # but handle if unicode is passed
        if isinstance(s, unicode):
            return s.encode(encoding)
        else:
            return s