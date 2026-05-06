def pinyin_syllable_to_zhuyin(s):
    """Convert Pinyin syllable *s* to a Zhuyin syllable."""
    pinyin_syllable, tone = _parse_pinyin_syllable(s)
    try:
        zhuyin_syllable = _PINYIN_MAP[pinyin_syllable.lower()]['Zhuyin']
    except KeyError:
        raise ValueError('Not a valid syllable: %s' % s)
    return zhuyin_syllable + _ZHUYIN_TONES[tone]