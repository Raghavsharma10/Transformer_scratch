def Bmatrix(C):
    """
    Calculate a matrix which is effectively the square root of the correlation matrix C


    Parameters
    ----------
    C : 2d array
        A covariance matrix

    Returns
    -------
    B : 2d array
        A matrix B such the B.dot(B') = inv(C)
    """
    # this version of finding the square root of the inverse matrix
    # suggested by Cath Trott
    L, Q = eigh(C)
    # force very small eigenvalues to have some minimum non-zero value
    minL = 1e-9*L[-1]
    L[L < minL] = minL
    S = np.diag(1 / np.sqrt(L))
    B = Q.dot(S)
    return B