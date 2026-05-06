def _parse_numbered_syllable(unparsed_syllable):
    """Return the syllable and tone of a numbered Pinyin syllable."""
    tone_number = unparsed_syllable[-1]
    if not tone_number.isdigit():
        syllable, tone = unparsed_syllable, '5'
    elif tone_number == '0':
        syllable, tone = unparsed_syllable[:-1], '5'
    elif tone_number in '12345':
        syllable, tone = unparsed_syllable[:-1], tone_number
    else:
        raise ValueError("Invalid syllable: %s" % unparsed_syllable)
    return syllable, tone