def interpolate2dStructuredFastIDW(grid, mask, kernel=15, power=2,
                                   minnvals=5):
    '''
    FASTER IMPLEMENTATION OF interpolate2dStructuredIDW

    replace all values in [grid] indicated by [mask]
    with the inverse distance weighted interpolation of all values within
    px+-kernel
    [power] -> distance weighting factor: 1/distance**[power]

    [minvals] -> minimum number of neighbour values to find until
                 interpolation stops

    '''
    indices, dist = growPositions(kernel)
    weights = 1 / dist**(0.5 * power)

    return _calc(grid, mask, indices, weights, minnvals - 1)