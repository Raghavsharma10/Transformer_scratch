def superpose(ras, rbs, weights=None):
    """Compute the transformation that minimizes the RMSD between the points ras and rbs

       Arguments:
        | ``ras``  --  a ``np.array`` with 3D coordinates of geometry A,
                       shape=(N,3)
        | ``rbs``  --  a ``np.array`` with 3D coordinates of geometry B,
                       shape=(N,3)

       Optional arguments:
        | ``weights``  --  a numpy array with fitting weights for each
                           coordinate, shape=(N,)

       Return value:
        | ``transformation``  --  the transformation that brings geometry A into
                                  overlap with geometry B

       Each row in ras and rbs represents a 3D coordinate. Corresponding rows
       contain the points that are brought into overlap by the fitting
       procedure. The implementation is based on the Kabsch Algorithm:

       http://dx.doi.org/10.1107%2FS0567739476001873
    """
    if weights is None:
        ma = ras.mean(axis=0)
        mb = rbs.mean(axis=0)
    else:
        total_weight = weights.sum()
        ma = np.dot(weights, ras)/total_weight
        mb = np.dot(weights, rbs)/total_weight


    # Kabsch
    if weights is None:
        A = np.dot((rbs-mb).transpose(), ras-ma)
    else:
        weights = weights.reshape((-1, 1))
        A = np.dot(((rbs-mb)*weights).transpose(), (ras-ma)*weights)
    v, s, wt = np.linalg.svd(A)
    s[:] = 1
    if np.linalg.det(np.dot(v, wt)) < 0:
        s[2] = -1
    r = np.dot(wt.T*s, v.T)
    return Complete(r, np.dot(r, -mb) + ma)