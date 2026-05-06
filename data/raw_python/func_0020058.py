def MaskSolve(A, b, w=5, progress=True, niter=None):
    '''
    Finds the solution `x` to the linear problem

        A x = b

    for all contiguous `w`-sized masks applied to
    the rows and columns of `A` and to the entries
    of `b`.

    Returns an array `X` of shape `(N - w + 1, N - w)`,
    where the `nth` row is the solution to the equation

        A[![n,n+w)] x = b[![n,n+w)]

    where ![n,n+w) indicates that indices in the range
    [n,n+w) have been masked.

    '''

    # Ensure we have choldate installed
    if cholupdate is None:
        log.info("Running the slow version of `MaskSolve`.")
        log.info("Install the `choldate` package for better performance.")
        log.info("https://github.com/rodluger/choldate")
        return MaskSolveSlow(A, b, w=w, progress=progress, niter=niter)

    # Number of data points
    N = b.shape[0]

    # How many iterations? Default is to go through
    # the entire dataset
    if niter is None:
        niter = N - w + 1

    # Our result matrix
    X = np.empty((niter, N - w))

    # Solve the first two steps explicitly.
    for n in range(2):
        mask = np.arange(n, w + n)
        A_ = np.delete(np.delete(A, mask, axis=0), mask, axis=1)
        b_ = np.delete(b, mask)
        U = cholesky(A_)
        X[n] = cho_solve((U, False), b_)

    # Iterate!
    for n in prange(1, niter - 1):

        # Update the data vector.
        b_[n] = b[n]

        # Remove a row.
        S33 = U[n + 1:, n + 1:]
        S23 = U[n, n + 1:]
        cholupdate(S33, S23)

        # Add a row.
        A12 = A[:n, n]
        A22 = A[n, n]
        A23 = A[n, n + w + 1:]
        S11 = U[:n, :n]
        S12 = solve_triangular(S11.T, A12, lower=True,
                               check_finite=False, trans=0, overwrite_b=True)
        S22 = np.sqrt(A22 - np.dot(S12.T, S12))
        S13 = U[:n, n + 1:]
        S23 = (A23 - np.dot(S12.T, S13)) / S22
        choldowndate(S33, np.array(S23))
        U[:n, n] = S12
        U[n, n] = S22
        U[n, n + 1:] = S23
        U[n + 1:, n + 1:] = S33

        # Now we can solve our linear equation
        X[n + 1] = cho_solve((U, False), b_)

    # Return the matrix
    return X