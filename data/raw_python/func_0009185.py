def to_ipa(s):
    """Convert *s* to IPA."""
    identity = identify(s)
    if identity == IPA:
        return s
    elif identity == PINYIN:
        return pinyin_to_ipa(s)
    elif identity == ZHUYIN:
        return zhuyin_to_ipa(s)
    else:
        raise ValueError("String is not a valid Chinese transcription.")