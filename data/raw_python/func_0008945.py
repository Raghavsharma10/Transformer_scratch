def get_border_index(I, shape, size):
    """
    Get flattened indices for the border of the region I.

    Parameters
    ----------
    I : np.ndarray(dtype=int)
        indices in the flattened region.
    size : int
        region size (technically computable from shape argument)
    shape : tuple(int, int)
        region shape

    Returns
    -------
    J : np.ndarray(dtype=int)
        indices orthogonally and diagonally bordering I
    """

    J = get_adjacent_index(I, shape, size)

    # instead of setdiff?
    # border = np.zeros(size)
    # border[J] = 1
    # border[I] = 0
    # J, = np.where(border)

    return np.setdiff1d(J, I)