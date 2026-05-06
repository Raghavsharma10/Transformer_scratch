def medfilt(vector, window):
    """
    Apply a window-length median filter to a 1D array vector.

    Should get rid of 'spike' value 15.
    >>> print(medfilt(np.array([1., 15., 1., 1., 1.]), 3))
    [1. 1. 1. 1. 1.]

    The 'edge' case is a bit tricky...
    >>> print(medfilt(np.array([15., 1., 1., 1., 1.]), 3))
    [15.  1.  1.  1.  1.]

    Inspired by: https://gist.github.com/bhawkins/3535131
    """
    if not window % 2 == 1:
        raise ValueError("Median filter length must be odd.")
    if not vector.ndim == 1:
        raise ValueError("Input must be one-dimensional.")

    k = (window - 1) // 2  # window movement
    result = np.zeros((len(vector), window), dtype=vector.dtype)
    result[:, k] = vector
    for i in range(k):
        j = k - i
        result[j:, i] = vector[:-j]
        result[:j, i] = vector[0]
        result[:-j, -(i + 1)] = vector[j:]
        result[-j:, -(i + 1)] = vector[-1]

    return np.median(result, axis=1)