def convert_ascii_field(string):
    """
    Convert an ASCII field into the corresponding list of Unicode strings.

    The (input) ASCII field is a Unicode string containing
    one or more ASCII codepoints (``00xx`` or ``U+00xx`` or
    an ASCII string not starting with ``00`` or ``U+``),
    separated by a space.

    :param str string: the (input) ASCII field
    :rtype: list of Unicode strings
    """
    values = []
    for codepoint in [s for s in string.split(DATA_FILE_CODEPOINT_SEPARATOR) if (s != DATA_FILE_VALUE_NOT_AVAILABLE) and (len(s) > 0)]:
        #if DATA_FILE_CODEPOINT_JOINER in codepoint:
        #    values.append(u"".join([hex_to_unichr(c) for c in codepoint.split(DATA_FILE_CODEPOINT_JOINER)]))
        if (codepoint.startswith(DATA_FILE_ASCII_NUMERICAL_CODEPOINT_START)) or (codepoint.startswith(DATA_FILE_ASCII_UNICODE_CODEPOINT_START)):
            values.append(hex_to_unichr(codepoint))
        else:
            values.append(codepoint)
    return values