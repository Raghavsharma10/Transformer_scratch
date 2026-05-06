def _numbered_vowel_to_accented(vowel, tone):
    """Convert a numbered Pinyin vowel to an accented Pinyin vowel."""
    if isinstance(tone, int):
        tone = str(tone)
    return _PINYIN_TONES[vowel + tone]