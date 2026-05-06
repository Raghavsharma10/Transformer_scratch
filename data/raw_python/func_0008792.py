def make_ita(noise, acf=None):
    """
    Create the matrix ita of the noise where the noise may be a masked array
    where ita(x,y) is the correlation between pixel pairs that have the same separation as x and y.

    Parameters
    ----------
    noise : 2d-array
        The noise image

    acf : 2d-array
        The autocorrelation matrix. (None = calculate from data).
        Default = None.

    Returns
    -------
    ita : 2d-array
        The matrix ita
    """
    if acf is None:
        acf = nan_acf(noise)
    # s should be the number of non-masked pixels
    s = np.count_nonzero(np.isfinite(noise))
    # the indices of the non-masked pixels
    xm, ym = np.where(np.isfinite(noise))
    ita = np.zeros((s, s))
    # iterate over the pixels
    for i, (x1, y1) in enumerate(zip(xm, ym)):
        for j, (x2, y2) in enumerate(zip(xm, ym)):
            k = abs(x1-x2)
            l = abs(y1-y2)
            ita[i, j] = acf[k, l]
    return ita