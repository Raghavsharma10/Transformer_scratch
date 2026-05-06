def _parse_zhuyin_syllable(unparsed_syllable):
    """Return the syllable and tone of a Zhuyin syllable."""
    zhuyin_tone = unparsed_syllable[-1]
    if zhuyin_tone in zhon.zhuyin.characters:
        syllable, tone = unparsed_syllable, '1'
    elif zhuyin_tone in zhon.zhuyin.marks:
        for tone_number, tone_mark in _ZHUYIN_TONES.items():
            if zhuyin_tone == tone_mark:
                syllable, tone = unparsed_syllable[:-1], tone_number
    else:
        raise ValueError("Invalid syllable: %s" % unparsed_syllable)

    return syllable, tone