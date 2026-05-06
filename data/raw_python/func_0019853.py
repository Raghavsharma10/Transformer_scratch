def spsolve(A, b):
    """Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The square matrix A will be converted into CSC or CSR form
    b : ndarray or sparse matrix
        The matrix or vector representing the right hand side of the equation.

    Returns
    -------
    x : ndarray or sparse matrix
        the solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[0]
        If b is a matrix, then x is a matrix of size (A.shape[0],)+b.shape[1:]

    """
    x = UmfpackLU(A).solve(b)

    if b.ndim == 2 and b.shape[1] == 1:
        # compatibility with scipy.sparse.spsolve quirk
        return x.ravel()
    else:
        return x