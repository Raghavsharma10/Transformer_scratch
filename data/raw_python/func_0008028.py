def aveknt(t, k):
    """Compute the running average of `k` successive elements of `t`. Return the averaged array.

Parameters:
    t:
        Python list or rank-1 array
    k:
        int, >= 2, how many successive elements to average

Returns:
    rank-1 array, averaged data. If k > len(t), returns a zero-length array.

Caveat:
    This is slightly different from MATLAB's aveknt, which returns the running average
    of `k`-1 successive elements of ``t[1:-1]`` (and the empty vector if  ``len(t) - 2 < k - 1``).

"""
    t = np.atleast_1d(t)
    if t.ndim > 1:
        raise ValueError("t must be a list or a rank-1 array")

    n = t.shape[0]
    u = max(0, n - (k-1))  # number of elements in the output array
    out = np.empty( (u,), dtype=t.dtype )

    for j in range(u):
        out[j] = sum( t[j:(j+k)] ) / k

    return out