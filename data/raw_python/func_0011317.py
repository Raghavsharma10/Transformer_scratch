def f2p_list(phrase, max_word_size=15, cutoff=3):
    """Convert a phrase from Finglish to Persian.

    phrase: The phrase to convert.

    max_word_size: Maximum size of the words to consider. Words larger
    than this will be kept unchanged.

    cutoff: The cut-off point. For each word, there could be many
    possibilities. By default 3 of these possibilities are considered
    for each word. This number can be changed by this argument.

    Returns a list of lists, each sub-list contains a number of
    possibilities for each word as a pair of (word, confidence)
    values.

    """

    # split the phrase into words
    results = [w for w in sep_regex.split(phrase) if w]

    # return an empty list if no words
    if results == []:
        return []

    # convert each word separately
    results = [f2p_word(w, max_word_size, cutoff) for w in results]

    return results