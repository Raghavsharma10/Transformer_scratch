def wald_wolfowitz(sequence):
    """
    implements the wald-wolfowitz runs test:
    http://en.wikipedia.org/wiki/Wald-Wolfowitz_runs_test
    http://support.sas.com/kb/33/092.html

    :param sequence: any iterable with at most 2 values. e.g.
                     '1001001'
                     [1, 0, 1, 0, 1]
                     'abaaabbba'

    :rtype: a dict with keys of
        `n_runs`: the number of runs in the sequence
        `p`: the support to reject the null-hypothesis that the number of runs
             supports a random sequence
        `z`: the z-score, used to calculate the p-value
        `sd`, `mean`: the expected standard deviation, mean the number of runs,
                      given the ratio of numbers of 1's/0's in the sequence

    >>> r = wald_wolfowitz('1000001')
    >>> r['n_runs'] # should be 3, because 1, 0, 1
    3

    >>> r['p'] < 0.05 # not < 0.05 evidence to reject Ho of random sequence
    False

    # this should show significance for non-randomness
    >>> li = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    >>> wald_wolfowitz(li)['p'] < 0.05
    True

    """
    R = n_runs = sum(1 for s in groupby(sequence, lambda a: a))

    n = float(sum(1 for s in sequence if s == sequence[0]))
    m = float(sum(1 for s in sequence if s != sequence[0]))

    # expected mean runs
    ER = ((2 * n * m ) / (n + m)) + 1
    # expected variance runs
    VR = (2 * n * m * (2 * n * m - n - m )) / ((n + m)**2 * (n + m - 1))
    O = (ER - 1) * (ER - 2) / (n + m - 1.)
    assert VR - O < 0.001, (VR, O)

    SD = math.sqrt(VR)
    # Z-score
    Z = (R - ER) / SD

    return {'z': Z, 'mean': ER, 'sd': SD, 'p': zprob(Z), 'n_runs': R}