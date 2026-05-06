def _ipa_syllable_to_numbered(s):
    """Convert IPA syllable *s* to a numbered Pinyin syllable."""
    ipa_syllable, tone = _parse_ipa_syllable(s)
    try:
        pinyin_syllable = _IPA_MAP[ipa_syllable]['Pinyin']
    except KeyError:
        raise ValueError('Not a valid syllable: %s' % s)
    return pinyin_syllable + tone