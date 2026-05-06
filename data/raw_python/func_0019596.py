def phyper(k, good, bad, N):
    """ Current hypergeometric implementation in scipy is broken, so here's the correct version """
    pvalues = [phyper_single(x, good, bad, N) for x in range(k + 1, N + 1)]
    return np.sum(pvalues)