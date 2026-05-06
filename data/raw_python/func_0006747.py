def _get_hangul_syllable_name(hangul_syllable):
    """
    Function for taking a Unicode scalar value representing a Hangul syllable and converting it to its syllable name as
    defined by the Unicode naming rule NR1.  See the Unicode Standard, ch. 04, section 4.8, Names, for more information.

    :param hangul_syllable: Unicode scalar value representing the Hangul syllable to convert
    :return: String representing its syllable name as transformed according to naming rule NR1.
    """
    if not _is_hangul_syllable(hangul_syllable):
        raise ValueError("Value passed in does not represent a Hangul syllable!")
    jamo = decompose_hangul_syllable(hangul_syllable, fully_decompose=True)
    result = ''
    for j in jamo:
        if j is not None:
            result += _get_jamo_short_name(j)
    return result