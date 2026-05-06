def pinyin_syllable_to_ipa(s):
    """Convert Pinyin syllable *s* to an IPA syllable."""
    pinyin_syllable, tone = _parse_pinyin_syllable(s)
    try:
        ipa_syllable = _PINYIN_MAP[pinyin_syllable.lower()]['IPA']
    except KeyError:
        raise ValueError('Not a valid syllable: %s' % s)
    return ipa_syllable + _IPA_TONES[tone]