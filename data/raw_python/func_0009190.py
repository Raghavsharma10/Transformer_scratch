def identify(s):
    """Identify a given string's transcription system.

    *s* is the string to identify. The string is checked to see if its
    contents are valid Pinyin, Zhuyin, or IPA. The :data:`PINYIN`,
    :data:`ZHUYIN`, and :data:`IPA` constants are returned to indicate the
    string's identity.
    If *s* is not a valid transcription system, then :data:`UNKNOWN` is
    returned.

    When checking for valid Pinyin or Zhuyin, testing is done on a syllable
    level, not a character level. For example, just because a string is
    composed of characters used in Pinyin, doesn't mean that it will identify
    as Pinyin; it must actually consist of valid Pinyin syllables. The same
    applies for Zhuyin.

    When checking for IPA, testing is only done on a character level. In other
    words, a string just needs to consist of Chinese IPA characters in order
    to identify as IPA.

    """
    if is_pinyin(s):
        return PINYIN
    elif is_zhuyin(s):
        return ZHUYIN
    elif is_ipa(s):
        return IPA
    else:
        return UNKNOWN