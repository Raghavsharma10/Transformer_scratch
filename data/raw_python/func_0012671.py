def kldiv(p, q, distp = None, distq = None, scale_factor = 1):
    """
    Computes the Kullback-Leibler divergence between two distributions.

    Parameters
        p : Matrix
            The first probability distribution
        q : Matrix
            The second probability distribution
        distp : fixmat
            If p is None, distp is used to compute a FDM which
            is then taken as 1st probability distribution.
        distq : fixmat
            If q is None, distq is used to compute a FDM which is
            then taken as 2dn probability distribution.
        scale_factor : double
            Determines the size of FDM computed from distq or distp.

    """
    assert q != None or distq != None, "Either q or distq have to be given"
    assert p != None or distp != None, "Either p or distp have to be given"

    try:
        if p == None:
            p = compute_fdm(distp, scale_factor = scale_factor)
        if q == None:
            q = compute_fdm(distq, scale_factor = scale_factor)
    except RuntimeError:
        return np.NaN

    q += np.finfo(q.dtype).eps
    p += np.finfo(p.dtype).eps
    kl = np.sum( p * (np.log2(p / q)))
    return kl