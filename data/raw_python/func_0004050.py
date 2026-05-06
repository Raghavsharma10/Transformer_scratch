def hex_to_unichr(hex_string):
    """
    Return the Unicode character with the given codepoint,
    given as an hexadecimal string.

    Return ``None`` if ``hex_string`` is ``None`` or is empty.

    Example::
        "0061"   => a
        "U+0061" => a

    :param str hex_string: the Unicode codepoint of the desired character
    :rtype: (Unicode) str
    """
    if (hex_string is None) or (len(hex_string) < 1):
        return None
    if hex_string.startswith("U+"):
        hex_string = hex_string[2:]
    return int_to_unichr(int(hex_string, base=16))