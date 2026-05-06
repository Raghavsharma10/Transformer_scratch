def init_uniform_ttable(wordlist):
    """
    Initialize (normalized) theta uniformly
    """
    n = len(wordlist)
    return numpy.ones((n, n + 1)) * (1 / n)