def estimate_k(X, max_k):
    """
    Estimate k for K-Means.

    Adapted from
    <https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/>
    """
    ks = range(1, max_k)
    fs = np.zeros(len(ks))

    # Special case K=1
    fs[0], Sk = _fK(1)

    # Rest of Ks
    for k in ks[1:]:
        fs[k-1], Sk = _fK(k, Skm1=Sk)
    return np.argmin(fs) + 1