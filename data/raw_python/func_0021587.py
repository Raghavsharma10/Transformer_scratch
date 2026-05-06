def ks_distance(a, b):
    '''Get the Kolmogorov-Smirnov (KS) distance between two densities a and b.'''
    if len(a.shape) == 1:
        return np.max(np.abs(a.cumsum() - b.cumsum()))
    return np.max(np.abs(a.cumsum(axis=1) - b.cumsum(axis=1)), axis=1)