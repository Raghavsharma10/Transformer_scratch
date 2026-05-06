def numbered_syllable_to_accented(s):
    """Convert numbered Pinyin syllable *s* to an accented Pinyin syllable.

    It implements the following algorithm to determine where to place tone
    marks:

    1. If the syllable has an 'a', 'e', or 'o' (in that order), put the
        tone mark over that vowel.
    2. Otherwise, put the tone mark on the last vowel.

    """
    if s == 'r5':
        return 'r'  # Special case for 'r' suffix.

    lowercase_syllable, case_memory = _lower_case(s)
    syllable, tone = _parse_numbered_syllable(lowercase_syllable)
    syllable = syllable.replace('v', '\u00fc')
    if re.search('[%s]' % _UNACCENTED_VOWELS, syllable) is None:
        return s
    if 'a' in syllable:
        accented_a = _numbered_vowel_to_accented('a', tone)
        accented_syllable = syllable.replace('a', accented_a)
    elif 'e' in syllable:
        accented_e = _numbered_vowel_to_accented('e', tone)
        accented_syllable = syllable.replace('e', accented_e)
    elif 'o' in syllable:
        accented_o = _numbered_vowel_to_accented('o', tone)
        accented_syllable = syllable.replace('o', accented_o)
    else:
        vowel = syllable[max(map(syllable.rfind, _UNACCENTED_VOWELS))]
        accented_vowel = _numbered_vowel_to_accented(vowel, tone)
        accented_syllable = syllable.replace(vowel, accented_vowel)
    return _restore_case(accented_syllable, case_memory)