def morphy_stem(word, pos=None):
    """
    Get the most likely stem for a word. If a part of speech is supplied,
    the stem will be more accurate.

    Valid parts of speech are:

    - 'n' or 'NN' for nouns
    - 'v' or 'VB' for verbs
    - 'a' or 'JJ' for adjectives
    - 'r' or 'RB' for adverbs

    Any other part of speech will be treated as unknown.
    """
    word = word.lower()
    if pos is not None:
        if pos.startswith('NN'):
            pos = 'n'
        elif pos.startswith('VB'):
            pos = 'v'
        elif pos.startswith('JJ'):
            pos = 'a'
        elif pos.startswith('RB'):
            pos = 'r'
    if pos is None and word.endswith('ing') or word.endswith('ed'):
        pos = 'v'
    if pos is not None and pos not in 'nvar':
        pos = None
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]
    if pos is None:
        if word in AMBIGUOUS_EXCEPTIONS:
            return AMBIGUOUS_EXCEPTIONS[word]
    return _morphy_best(word, pos) or word