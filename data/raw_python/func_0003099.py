def get_initial(s, delimiter=' '):
    """Return the 1st char of pinyin of string, the string must be unicode
    """
    initials = (p[0] for p in _pinyin_generator(u(s), format="strip"))
    return delimiter.join(initials)