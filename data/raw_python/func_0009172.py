def accented_syllable_to_numbered(s):
    """Convert accented Pinyin syllable *s* to a numbered Pinyin syllable."""
    if s[0] == '\u00B7':
        lowercase_syllable, case_memory = _lower_case(s[1:])
        lowercase_syllable = '\u00B7' + lowercase_syllable
    else:
        lowercase_syllable, case_memory = _lower_case(s)
    numbered_syllable, tone = _parse_accented_syllable(lowercase_syllable)
    return _restore_case(numbered_syllable, case_memory) + tone