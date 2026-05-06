def _get_jamo_short_name(jamo):
    """
    Function for taking a Unicode scalar value representing a Jamo and determining the correct value for its
    Jamo_Short_Name property.  For more information on the Jamo_Short_Name property see the Unicode Standard,
    ch. 03, section 3.12, Conjoining Jamo Behavior.

    https://www.unicode.org/versions/latest/ch03.pdf

    :param jamo: Unicode scalar value representing a Jamo
    :return: Returns a string representing its Jamo_Short_Name property
    """
    if not _is_jamo(jamo):
        raise ValueError("Value 0x%0.4x passed in does not represent a Jamo!" % jamo)
    if not _jamo_short_names:
        _load_jamo_short_names()
    return _jamo_short_names[jamo]