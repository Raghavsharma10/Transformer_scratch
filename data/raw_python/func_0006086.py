def random_words_string(count=1, maxchars=None, sep=''):
    """Gets a
    """
    nouns = sep.join([random_word() for x in xrange(0, count)])

    if maxchars is not None and nouns > maxchars:
        nouns = nouns[0:maxchars-1]

    return nouns