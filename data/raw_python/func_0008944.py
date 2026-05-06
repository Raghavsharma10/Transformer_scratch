def get_adjacent_index(I, shape, size):
    """
    Find indices 2d-adjacent to those in I. Helper function for get_border*.

    Parameters
    ----------
    I : np.ndarray(dtype=int)
        indices in the flattened region
    shape : tuple(int, int)
        region shape
    size : int
        region size (technically computable from shape)

    Returns
    -------
    J : np.ndarray(dtype=int)
        indices orthogonally and diagonally adjacent to I

    """

    m, n = shape
    In = I % n
    bL = In != 0
    bR = In != n-1
    
    J = np.concatenate([
        # orthonally adjacent
        I - n,
        I[bL] - 1,
        I[bR] + 1,
        I + n,

        # diagonally adjacent
        I[bL] - n-1,
        I[bR] - n+1,
        I[bL] + n-1,
        I[bR] + n+1])

    # remove indices outside the array
    J = J[(J>=0) & (J<size)]

    return J