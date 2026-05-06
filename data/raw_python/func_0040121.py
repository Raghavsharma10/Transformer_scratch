def init_perfect_ttable(words):
    """initialize (normalized) theta according to whether words rhyme"""
    d = read_celex()

    not_in_dict = 0

    n = len(words)
    t_table = numpy.zeros((n, n + 1))

    # initialize P(c|r) accordingly
    for r, w in enumerate(words):
        if w not in d:
            not_in_dict += 1
        for c, v in enumerate(words):
            if c < r:
                t_table[r, c] = t_table[c, r]
            elif w in d and v in d:
                t_table[r, c] = int(is_rhyme(d, w, v)) + 0.001  # for backoff
            else:
                t_table[r, c] = random.random()
        t_table[r, n] = random.random()  # no estimate for P(r|no history)

    print(not_in_dict, "of", n, " words are not in CELEX")

    # normalize
    for c in range(n + 1):
        tot = sum(t_table[:, c])
        for r in range(n):
            t_table[r, c] = t_table[r, c] / tot

    return t_table