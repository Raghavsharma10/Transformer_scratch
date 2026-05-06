def make_ascii(word):
    """
    Converts unicode-specific characters to their equivalent ascii
    """
    if sys.version_info < (3, 0, 0):
        word = unicode(word)
    else:
        word = str(word)

    normalized = unicodedata.normalize('NFKD', word)

    return normalized.encode('ascii', 'ignore').decode('utf-8')