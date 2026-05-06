def to_pinyin(s, accented=True):
    """Convert *s* to Pinyin.

    If *accented* is ``True``, diacritics are added to the Pinyin syllables. If
    it's ``False``, numbers are used to indicate tone.

    """
    identity = identify(s)
    if identity == PINYIN:
        if _has_accented_vowels(s):
            return s if accented else accented_to_numbered(s)
        else:
            return numbered_to_accented(s) if accented else s
    elif identity == ZHUYIN:
        return zhuyin_to_pinyin(s, accented=accented)
    elif identity == IPA:
        return ipa_to_pinyin(s, accented=accented)
    else:
        raise ValueError("String is not a valid Chinese transcription.")