def _get_hangul_syllable_type(hangul_syllable):
    """
    Function for taking a Unicode scalar value representing a Hangul syllable and determining the correct value for its
    Hangul_Syllable_Type property.  For more information on the Hangul_Syllable_Type property see the Unicode Standard,
    ch. 03, section 3.12, Conjoining Jamo Behavior.

    https://www.unicode.org/versions/latest/ch03.pdf

    :param hangul_syllable: Unicode scalar value representing a Hangul syllable
    :return: Returns a string representing its Hangul_Syllable_Type property ("L", "V", "T", "LV" or "LVT")
    """
    if not _is_hangul_syllable(hangul_syllable):
        raise ValueError("Value 0x%0.4x does not represent a Hangul syllable!" % hangul_syllable)
    if not _hangul_syllable_types:
        _load_hangul_syllable_types()
    return _hangul_syllable_types[hangul_syllable]