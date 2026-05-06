def interpolate2dStructuredPointSpreadIDW(grid, mask, kernel=15, power=2,
                                          maxIter=1e5, copy=True):
    '''
    same as interpolate2dStructuredIDW but using the point spread method
    this is faster if there are bigger connected masked areas and the border
    length is smaller

    replace all values in [grid] indicated by [mask]
    with the inverse distance weighted interpolation of all values within
    px+-kernel

    [power] -> distance weighting factor: 1/distance**[power]
    [copy] -> False: a bit faster, but modifies 'grid' and 'mask'
    '''
    assert grid.shape == mask.shape, 'grid and mask shape are different'

    border = np.zeros(shape=mask.shape, dtype=np.bool)
    if copy:
        # copy mask as well because if will be modified later:
        mask = mask.copy()
        grid = grid.copy()
    return _calc(grid, mask, border, kernel, power, maxIter)