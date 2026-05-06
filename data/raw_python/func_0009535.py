def interpolate2dStructuredCrossAvg(grid, mask, kernel=15, power=2):
    '''
    #######
    usefull if large empty areas need to be filled

    '''

    vals = np.empty(shape=4, dtype=grid.dtype)
    dist = np.empty(shape=4, dtype=np.uint16)
    weights = np.empty(shape=4, dtype=np.float32)
    valid = np.empty(shape=4, dtype=bool)

    return _calc(grid, mask, power, kernel, vals, dist, weights, valid)