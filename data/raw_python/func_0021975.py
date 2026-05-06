def auto_correlation(sequence):
    """
    test for the autocorrelation of a sequence between t and t - 1
    as the 'auto_correlation' it is less likely that the sequence is
    generated randomly.
    :param sequence: any iterable with at most 2 values that can be turned
                     into a float via np.float . e.g.
                     '1001001'
                     [1, 0, 1, 0, 1]
                     [1.2,.1,.5,1]
    :rtype: returns a dict of the linear regression stats of sequence[1:] vs.
            sequence[:-1]

    >>> result = auto_correlation('00000001111111111100000000')
    >>> result['p'] < 0.05
    True
    >>> result['auto_correlation']
    0.83766233766233755

    """
    if isinstance(sequence, basestring):
        sequence = map(int, sequence)
    seq = np.array(list(sequence), dtype=np.float)
    dseq = np.column_stack((seq[1:], seq[:-1]))
    slope, intercept, r, ttp, see = linregress(seq[1:], seq[:-1])
    cc = np.corrcoef(dseq, rowvar=0)[0][1]
    return {'slope': slope, 'intercept': intercept, 'r-squared': r ** 2,
            'p': ttp, 'see': see, 'auto_correlation': cc}