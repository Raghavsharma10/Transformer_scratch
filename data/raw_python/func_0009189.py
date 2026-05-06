def is_ipa(s):
    """Check if *s* consists of valid Chinese IPA."""
    re_pattern = ('(?:%(syllable)s|[ \t%(punctuation)s])+' %
                  {'syllable': _IPA_SYLLABLE,
                   'punctuation': zhon.pinyin.punctuation})
    return _is_pattern_match(re_pattern, s)