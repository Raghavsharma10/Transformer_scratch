def get(s, delimiter='', format="diacritical"):
    """Return pinyin of string, the string must be unicode
    """
    return delimiter.join(_pinyin_generator(u(s), format=format))