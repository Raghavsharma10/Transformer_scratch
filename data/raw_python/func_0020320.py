def _range10_90(x):
    '''
    Returns the 10th-90th percentile range of array :py:obj:`x`.

    '''

    x = np.delete(x, np.where(np.isnan(x)))
    i = np.argsort(x)
    a = int(0.1 * len(x))
    b = int(0.9 * len(x))
    return x[i][b] - x[i][a]