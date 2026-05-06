def _zhuyin_syllable_to_numbered(s):
    """Convert Zhuyin syllable *s* to a numbered Pinyin syllable."""
    zhuyin_syllable, tone = _parse_zhuyin_syllable(s)
    try:
        pinyin_syllable = _ZHUYIN_MAP[zhuyin_syllable]['Pinyin']
    except KeyError:
        raise ValueError('Not a valid syllable: %s' % s)
    return pinyin_syllable + tone