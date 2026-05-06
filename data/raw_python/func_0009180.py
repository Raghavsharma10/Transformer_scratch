def pinyin_to_ipa(s):
    """Convert all Pinyin syllables in *s* to IPA.

    Spaces are added between connected syllables and syllable-separating
    apostrophes are removed.

    """
    return _convert(s, zhon.pinyin.syllable, pinyin_syllable_to_ipa,
                    remove_apostrophes=True, separate_syllables=True)