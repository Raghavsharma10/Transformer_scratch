def unescape(s):
    r"""Inverse of `escape`.
    >>> unescape(r'\x41\n\x42\n\x43')
    'A\nB\nC'
    >>> unescape(r'\u86c7')
    u'\u86c7'
    >>> unescape(u'ah')
    u'ah'
    """
    if re.search(r'(?<!\\)\\(\\\\)*[uU]', s) or isinstance(s, unicode):
        return unescapeUnicode(s)
    else:
        return unescapeAscii(s)