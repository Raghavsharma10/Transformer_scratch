def pairwise_ellpitical_binary(sources, eps, far=None):
    """
    Do a pairwise comparison of all sources and determine if they have a normalized distance within
    eps.

    Form this into a matrix of shape NxN.


    Parameters
    ----------
    sources : list
        A list of sources (objects with parameters: ra,dec,a,b,pa)

    eps : float
        Normalised distance constraint.

    far : float
        If sources have a dec that differs by more than this amount then they are considered to be not matched.
        This is a short-cut around performing GCD calculations.

    Returns
    -------
    prob : numpy.ndarray
        A 2d array of True/False.

    See Also
    --------
    :func:`AegeanTools.cluster.norm_dist`
    """
    if far is None:
        far = max(a.a/3600 for a in sources)
    l = len(sources)
    distances = np.zeros((l, l), dtype=bool)
    for i in range(l):
        for j in range(i, l):
            if i == j:
                distances[i, j] = False
                continue
            src1 = sources[i]
            src2 = sources[j]
            if src2.dec - src1.dec > far:
                break
            if abs(src2.ra - src1.ra)*np.cos(np.radians(src1.dec)) > far:
                continue
            distances[i, j] = norm_dist(src1, src2) > eps
            distances[j, i] = distances[i, j]
    return distances