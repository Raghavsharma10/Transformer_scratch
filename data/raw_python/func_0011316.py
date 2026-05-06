def f2p_word(word, max_word_size=15, cutoff=3):
    """Convert a single word from Finglish to Persian.

    max_word_size: Maximum size of the words to consider. Words larger
    than this will be kept unchanged.

    cutoff: The cut-off point. For each word, there could be many
    possibilities. By default 3 of these possibilities are considered
    for each word. This number can be changed by this argument.

    """

    original_word = word
    word = word.lower()

    c = dictionary.get(word)
    if c:
        return [(c, 1.0)]

    if word == '':
        return []
    elif len(word) > max_word_size:
        return [(original_word, 1.0)]

    results = []
    for w in variations(word):
        results.extend(f2p_word_internal(w, original_word))

    # sort results based on the confidence value
    results.sort(key=lambda r: r[1], reverse=True)

    # return the top three results in order to cut down on the number
    # of possibilities.
    return results[:cutoff]