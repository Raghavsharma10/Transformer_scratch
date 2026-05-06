def pinyin_to_zhuyin(s):
    """Convert all Pinyin syllables in *s* to Zhuyin.

    Spaces are added between connected syllables and syllable-separating
    apostrophes are removed.

    """
    return _convert(s, zhon.pinyin.syllable, pinyin_syllable_to_zhuyin,
                    remove_apostrophes=True, separate_syllables=True)