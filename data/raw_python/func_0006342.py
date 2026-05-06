def in1d_sorted(ar1, ar2):
    """
    Does the same than np.in1d but uses the fact that ar1 and ar2 are sorted. Is therefore much faster.

    """
    if ar1.shape[0] == 0 or ar2.shape[0] == 0:  # check for empty arrays to avoid crash
        return []
    inds = ar2.searchsorted(ar1)
    inds[inds == len(ar2)] = 0
    return ar2[inds] == ar1