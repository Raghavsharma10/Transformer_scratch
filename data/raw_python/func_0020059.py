def MaskSolveSlow(A, b, w=5, progress=True, niter=None):
    '''
    Identical to `MaskSolve`, but computes the solution
    the brute-force way.

    '''

    # Number of data points
    N = b.shape[0]

    # How many iterations? Default is to go through
    # the entire dataset
    if niter is None:
        niter = N - w + 1

    # Our result matrix
    X = np.empty((niter, N - w))

    # Iterate! The mask at step `n` goes from
    # data index `n` to data index `n+w-1` (inclusive).
    for n in prange(niter):
        mask = np.arange(n, n + w)
        An = np.delete(np.delete(A, mask, axis=0), mask, axis=1)
        Un = cholesky(An)
        bn = np.delete(b, mask)
        X[n] = cho_solve((Un, False), bn)

    return X