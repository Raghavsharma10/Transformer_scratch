def numbered_to_accented(s):
    """Convert all numbered Pinyin syllables in *s* to accented Pinyin."""
    return _convert(s, zhon.pinyin.syllable, numbered_syllable_to_accented,
                    add_apostrophes=True)