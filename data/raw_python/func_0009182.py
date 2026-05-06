def ipa_to_pinyin(s, accented=True):
    """Convert all IPA syllables in *s* to Pinyin.

    If *accented* is ``True``, diacritics are added to the Pinyin syllables. If
    it's ``False``, numbers are used to indicate tone.

    """
    if accented:
        function = _ipa_syllable_to_accented
    else:
        function = _ipa_syllable_to_numbered
    return _convert(s, _IPA_SYLLABLE, function)