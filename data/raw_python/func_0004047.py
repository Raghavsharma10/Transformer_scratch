def is_unicode_string(string):
    """
    Return ``True`` if the given string is a Unicode string,
    that is, of type ``unicode`` in Python 2 or ``str`` in Python 3.

    Return ``None`` if ``string`` is ``None``.

    :param str string: the string to be checked
    :rtype: bool
    """
    if string is None:
        return None
    if PY2:
        return isinstance(string, unicode)
    return isinstance(string, str)