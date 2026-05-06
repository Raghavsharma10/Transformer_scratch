def to_zhuyin(s):
    """Convert *s* to Zhuyin."""
    identity = identify(s)
    if identity == ZHUYIN:
        return s
    elif identity == PINYIN:
        return pinyin_to_zhuyin(s)
    elif identity == IPA:
        return ipa_to_zhuyin(s)
    else:
        raise ValueError("String is not a valid Chinese transcription.")