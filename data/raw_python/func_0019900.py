def chol(A):
    """
    Calculate the lower triangular matrix of the Cholesky decomposition of
    a symmetric, positive-definite matrix.
    """
    A = np.array(A)
    assert A.shape[0] == A.shape[1], "Input matrix must be square"

    L = [[0.0] * len(A) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (
                (A[i][i] - s) ** 0.5 if (i == j) else (1.0 / L[j][j] * (A[i][j] - s))
            )

    return np.array(L)