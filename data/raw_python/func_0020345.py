def lnlike(x, star):
    """Return the log likelihood given parameter vector `x`."""
    ll = lnprior(x)
    if np.isinf(ll):
        return ll, (np.nan, np.nan)
    per, t0, b = x
    model = TransitModel('b', per=per, t0=t0, b=b, rhos=10.)(star.time)
    like, d, vard = star.lnlike(model, full_output=True)
    ll += like
    return ll, (d,)