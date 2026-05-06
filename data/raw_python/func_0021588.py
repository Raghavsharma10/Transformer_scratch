def tv_distance(a, b):
    '''Get the Total Variation (TV) distance between two densities a and b.'''
    if len(a.shape) == 1:
        return np.sum(np.abs(a - b))
    return np.sum(np.abs(a - b), axis=1)