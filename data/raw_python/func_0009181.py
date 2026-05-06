def zhuyin_to_pinyin(s, accented=True):
    """Convert all Zhuyin syllables in *s* to Pinyin.

    If *accented* is ``True``, diacritics are added to the Pinyin syllables. If
    it's ``False``, numbers are used to indicate tone.

    """
    if accented:
        function = _zhuyin_syllable_to_accented
    else:
        function = _zhuyin_syllable_to_numbered
    return _convert(s, zhon.zhuyin.syllable, function)