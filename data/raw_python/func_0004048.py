def to_unicode_string(string):
    """
    Return a Unicode string out of the given string.
    
    On Python 2, it calls ``unicode`` with ``utf-8`` encoding.
    On Python 3, it just returns the given string.

    Return ``None`` if ``string`` is ``None``.

    :param str string: the string to convert to Unicode
    :rtype: (Unicode) str
    """
    if string is None:
        return None
    if is_unicode_string(string):
        return string
    # if reached here, string is a byte string 
    if PY2:
        return unicode(string, encoding="utf-8")
    return string.decode(encoding="utf-8")