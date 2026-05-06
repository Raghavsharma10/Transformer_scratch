def identify(s):
    """Identify what kind of Chinese characters a string contains.

    *s* is a string to examine. The string's Chinese characters are tested to
    see if they are compatible with the Traditional or Simplified characters
    systems, compatible with both, or contain a mixture of Traditional and
    Simplified characters. The :data:`TRADITIONAL`, :data:`SIMPLIFIED`,
    :data:`BOTH`, or :data:`MIXED` constants are returned to indicate the
    string's identity. If *s* contains no Chinese characters, then
    :data:`UNKNOWN` is returned.

    All characters in a string that aren't found in the CC-CEDICT dictionary
    are ignored.

    Because the Traditional and Simplified Chinese character systems overlap, a
    string containing Simplified characters could identify as
    :data:`SIMPLIFIED` or :data:`BOTH` depending on if the characters are also
    Traditional characters. To make testing the identity of a string easier,
    the functions :func:`is_traditional`, :func:`is_simplified`, and
    :func:`has_chinese` are provided.

    """
    chinese = _get_hanzi(s)
    if not chinese:
        return UNKNOWN
    if chinese.issubset(_SHARED_CHARACTERS):
        return BOTH
    if chinese.issubset(_TRADITIONAL_CHARACTERS):
        return TRADITIONAL
    if chinese.issubset(_SIMPLIFIED_CHARACTERS):
        return SIMPLIFIED
    return MIXED