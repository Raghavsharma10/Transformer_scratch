def _normalize(x, cmin=None, cmax=None, clip=True):
    """Normalize an array from the range [cmin, cmax] to [0,1],
    with optional clipping."""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if cmin is None:
        cmin = x.min()
    if cmax is None:
        cmax = x.max()
    if cmin == cmax:
        return .5 * np.ones(x.shape)
    else:
        cmin, cmax = float(cmin), float(cmax)
        y = (x - cmin) * 1. / (cmax - cmin)
        if clip:
            y = np.clip(y, 0., 1.)
        return y