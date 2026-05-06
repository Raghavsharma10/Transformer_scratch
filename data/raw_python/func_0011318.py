def f2p(phrase, max_word_size=15, cutoff=3):
    """Convert a Finglish phrase to the most probable Persian phrase.

    """

    results = f2p_list(phrase, max_word_size, cutoff)
    return ' '.join(i[0][0] for i in results)