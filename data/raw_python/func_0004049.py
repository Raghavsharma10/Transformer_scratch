def to_str(string):
    """
    Return the given string (either byte string or Unicode string)
    converted to native-str, that is,
    a byte string on Python 2, or a Unicode string on Python 3.

    Return ``None`` if ``string`` is ``None``.

    :param str string: the string to convert to native-str
    :rtype: native-str
    """
    if string is None:
        return None
    if isinstance(string, str):
        return string
    if PY2:
        return string.encode("utf-8")
    return string.decode("utf-8")