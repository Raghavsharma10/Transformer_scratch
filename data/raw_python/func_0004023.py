def convert_unicode_field(string):
    """
    Convert a Unicode field into the corresponding list of Unicode strings.

    The (input) Unicode field is a Unicode string containing
    one or more Unicode codepoints (``xxxx`` or ``U+xxxx`` or ``xxxx_yyyy``),
    separated by a space.

    :param str string: the (input) Unicode field
    :rtype: list of Unicode strings
    """
    values = []
    for codepoint in [s for s in string.split(DATA_FILE_CODEPOINT_SEPARATOR) if (s != DATA_FILE_VALUE_NOT_AVAILABLE) and (len(s) > 0)]:
        values.append(u"".join([hex_to_unichr(c) for c in codepoint.split(DATA_FILE_CODEPOINT_JOINER)]))
    return values