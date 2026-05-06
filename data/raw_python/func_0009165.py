def _accented_vowel_to_numbered(vowel):
    """Convert an accented Pinyin vowel to a numbered Pinyin vowel."""
    for numbered_vowel, accented_vowel in _PINYIN_TONES.items():
        if vowel == accented_vowel:
            return tuple(numbered_vowel)