def preservesurrogates(s):
    """
    Function for splitting a string into a list of characters, preserving surrogate pairs.

    In python 2, unicode characters above 0x10000 are stored as surrogate pairs.  For example, the Unicode character
    u"\U0001e900" is stored as the surrogate pair u"\ud83a\udd00":

    s = u"AB\U0001e900CD"
    len(s) -> 6
    list(s) -> [u'A', u'B', u'\ud83a', u'\udd00', u'C', 'D']
    len(preservesurrogates(s)) -> 5
    list(preservesurrogates(s)) -> [u'A', u'B', u'\U0001e900', u'C', u'D']

    :param s: String to split
    :return: List of characters
    """
    if not isinstance(s, six.text_type):
        raise TypeError(u"String to split must be of type 'unicode'!")
    surrogates_regex_str = u"[{0}-{1}][{2}-{3}]".format(HIGH_SURROGATE_START,
                                                        HIGH_SURROGATE_END,
                                                        LOW_SURROGATE_START,
                                                        LOW_SURROGATE_END)
    surrogates_regex = re.compile(u"(?:{0})|.".format(surrogates_regex_str))
    return surrogates_regex.findall(s)