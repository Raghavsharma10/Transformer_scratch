def _parse_accented_syllable(unparsed_syllable):
    """Return the syllable and tone of an accented Pinyin syllable.

    Any accented vowels are returned without their accents.

    Implements the following algorithm:

    1. If the syllable has an accent mark, convert that vowel to a
        regular vowel and add the tone to the end of the syllable.
    2. Otherwise, assume the syllable is tone 5 (no accent marks).

    """
    if unparsed_syllable[0] == '\u00B7':
        # Special case for middle dot tone mark.
        return unparsed_syllable[1:], '5'
    for character in unparsed_syllable:
        if character in _ACCENTED_VOWELS:
            vowel, tone = _accented_vowel_to_numbered(character)
            return unparsed_syllable.replace(character, vowel), tone
    return unparsed_syllable, '5'