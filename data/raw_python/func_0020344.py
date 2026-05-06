def lnprior(x):
    """Return the log prior given parameter vector `x`."""
    per, t0, b = x
    if b < -1 or b > 1:
        return -np.inf
    elif per < 7 or per > 10:
        return -np.inf
    elif t0 < 1978 or t0 > 1979:
        return -np.inf
    else:
        return 0.