def get_distance(region, src):
    """
    Compute within-region distances from the src pixels.

    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype=bool)
        mask of the region
    src : np.ndarray(shape=(m, n), dtype=bool)
        mask of the source pixels to compute distances from.

    Returns
    -------
    d : np.ndarray(shape=(m, n), dtype=float)
        approximate within-region distance from the nearest src pixel;
        (distances outside of the region are arbitrary).
    """

    dmax = float(region.size)
    d = np.full(region.shape, dmax)
    d[src] = 0
    for n in range(region.size):
        d_orth = minimum_filter(d, footprint=_ORTH2) + 1
        d_diag = minimum_filter(d, (3, 3)) + _SQRT2
        d_adj = np.minimum(d_orth[region], d_diag[region])
        d[region] = np.minimum(d_adj, d[region])
        if (d[region] < dmax).all():
            break
    return d