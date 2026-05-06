def init_distance_ttable(wordlist, distance_function):
    """
    Initialize pair-wise rhyme strenghts according to the given word distance function
    """
    n = len(wordlist)
    t_table = numpy.zeros((n, n + 1))

    # Initialize P(c|r) accordingly
    for r, w in enumerate(wordlist):
        for c, v in enumerate(wordlist):
            if c < r:
                t_table[r, c] = t_table[c, r]  # Similarity is symmetric
            else:
                t_table[r, c] = distance_function(w, v) + 0.001  # For backoff
    t_table[:, n] = numpy.mean(t_table[:, :-1], axis=1)

    # Normalize
    t_totals = numpy.sum(t_table, axis=0)
    for i, t_total in enumerate(t_totals.tolist()):
        t_table[:, i] /= t_total
    return t_table