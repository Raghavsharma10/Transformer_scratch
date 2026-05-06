def unique_rows(arr, return_index=False, return_inverse=False):
    """Returns a copy of arr with duplicate rows removed.
    
    From Stackoverflow "Find unique rows in numpy.array."
    
    Parameters
    ----------
    arr : :py:class:`Array`, (`m`, `n`)
        The array to find the unique rows of.
    return_index : bool, optional
        If True, the indices of the unique rows in the array will also be
        returned. I.e., unique = arr[idx]. Default is False (don't return
        indices).
    return_inverse: bool, optional
        If True, the indices in the unique array to reconstruct the original
        array will also be returned. I.e., arr = unique[inv]. Default is False
        (don't return inverse).
    
    Returns
    -------
    unique : :py:class:`Array`, (`p`, `n`) where `p` <= `m`
        The array `arr` with duplicate rows removed.
    """
    b = scipy.ascontiguousarray(arr).view(
        scipy.dtype((scipy.void, arr.dtype.itemsize * arr.shape[1]))
    )
    try:
        out = scipy.unique(b, return_index=True, return_inverse=return_inverse)
        dum = out[0]
        idx = out[1]
        if return_inverse:
            inv = out[2]
    except TypeError:
        if return_inverse:
            raise RuntimeError(
                "Error in scipy.unique on older versions of numpy prevents "
                "return_inverse from working!"
            )
        # Handle bug in numpy 1.6.2:
        rows = [_Row(row) for row in b]
        srt_idx = sorted(range(len(rows)), key=rows.__getitem__)
        rows = scipy.asarray(rows)[srt_idx]
        row_cmp = [-1]
        for k in xrange(1, len(srt_idx)):
            row_cmp.append(rows[k-1].__cmp__(rows[k]))
        row_cmp = scipy.asarray(row_cmp)
        transition_idxs = scipy.where(row_cmp != 0)[0]
        idx = scipy.asarray(srt_idx)[transition_idxs]
    out = arr[idx]
    if return_index:
        out = (out, idx)
    elif return_inverse:
        out = (out, inv)
    elif return_index and return_inverse:
        out = (out, idx, inv)
    return out