def get_border_mask(region):
    """
    Get border of the region as a boolean array mask.

    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype=bool)
        mask of the region

    Returns
    -------
    border : np.ndarray(shape=(m, n), dtype=bool)
        mask of the region border (not including region)
    """

    # common special case (for efficiency)
    internal = region[1:-1, 1:-1]
    if internal.all() and internal.any():
        return ~region
    
    I, = np.where(region.ravel())
    J = get_adjacent_index(I, region.shape, region.size)

    border = np.zeros(region.size, dtype='bool')
    border[J] = 1
    border[I] = 0
    border = border.reshape(region.shape)

    return border