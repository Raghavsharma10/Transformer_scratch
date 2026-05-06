def is_pinyin(s):
    """Check if *s* consists of valid Pinyin."""
    re_pattern = ('(?:%(word)s|[ \t%(punctuation)s])+' %
                  {'word': zhon.pinyin.word,
                   'punctuation': zhon.pinyin.punctuation})
    return _is_pattern_match(re_pattern, s)