def normalize_list(text):
    """
    Get a list of word stems that appear in the text. Stopwords and an initial
    'to' will be stripped, unless this leaves nothing in the stem.

    >>> normalize_list('the dog')
    ['dog']
    >>> normalize_list('big dogs')
    ['big', 'dog']
    >>> normalize_list('the')
    ['the']
    """
    pieces = [morphy_stem(word) for word in tokenize(text)]
    pieces = [piece for piece in pieces if good_lemma(piece)]
    if not pieces:
        return [text]
    if pieces[0] == 'to':
        pieces = pieces[1:]
    return pieces