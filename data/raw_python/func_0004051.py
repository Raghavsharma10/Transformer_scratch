def unicode_to_hex(unicode_string):
    """
    Return a string containing the Unicode hexadecimal codepoint
    of each Unicode character in the given Unicode string.

    Return ``None`` if ``unicode_string`` is ``None``.

    Example::
        a  => U+0061
        ab => U+0061 U+0062

    :param str unicode_string: the Unicode string to convert
    :rtype: (Unicode) str
    """
    if unicode_string is None:
        return None
    acc = []
    for c in unicode_string:
        s = hex(ord(c)).replace("0x", "").upper()
        acc.append("U+" + ("0" * (4 - len(s))) + s)
    return u" ".join(acc)