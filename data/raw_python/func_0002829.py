def fnd_unq_rws(A, return_index=False, return_inverse=False):
    """Find unique rows in 2D array.

    Parameters
    ----------
    A : 2d numpy array
        Array for which unique rows should be identified.
    return_index : bool
        Bool to decide whether I is returned.
    return_inverse : bool
        Bool to decide whether J is returned.

    Returns
    -------
    B : 1d numpy array,
        Unique rows
    I: 1d numpy array, only returned if return_index is True
        B = A[I,:]
    J: 2d numpy array, only returned if return_inverse is True
        A = B[J,:]

    """

    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                  return_index=return_index,
                  return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')