def solve_chol(A,B):
    """
    Solve cholesky decomposition::

        return A\(A'\B)

    """
    # X = linalg.solve(A,linalg.solve(A.transpose(),B))
    # much faster version
    X = linalg.cho_solve((A, True), B)
    return X