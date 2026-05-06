def _t(unistr, charset_from, charset_to):
    """
        This is a unexposed function, is responsibility for translation internal.
    """
    # if type(unistr) is str:
    #     try:
    #         unistr = unistr.decode('utf-8')
    #     # Python 3 returns AttributeError when .decode() is called on a str
    #     # This means it is already unicode.
    #     except AttributeError:
    #         pass
    # try:
    #     if type(unistr) is not unicode:
    #         return unistr
    # # Python 3 returns NameError because unicode is not a type.
    # except NameError:
    #     pass

    chars = []
    for c in unistr:
        idx = charset_from.find(c)
        chars.append(charset_to[idx] if idx!=-1 else c)
    return u''.join(chars)