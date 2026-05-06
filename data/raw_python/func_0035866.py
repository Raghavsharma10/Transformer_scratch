def peak_model(f):
    """
    Given a function that models a peak, add scale and location arguments to

    For all functions, v is vertical offset, h is height
    x is horizontal offset (1st moment), w is width (2nd moment),
    s is skewness (3rd moment), e is excess (4th moment)
    """
    @wraps(f)
    def wrapped_f(t, **kw):
        # load kwargs with default values
        # do this here instead of in the def because we want to parse
        # all of kwargs later to copy values to pass into f
        def_vals = {'v': 0.0, 'h': 1.0, 'x': 0.0, 'w': 1.0, 's': 1.1, 'e': 1.0}
        for v in def_vals:
            if v not in kw:
                kw[v] = def_vals[v]

        # this copies all of the defaults into what the peak function needs
        anames, _, _, _ = inspect.getargspec(f)
        fkw = dict([(arg, kw[arg]) for arg in anames if arg in kw])

        # some functions use location or width parameters explicitly
        # if not, adjust the timeseries accordingly
        ta = t
        if 'x' not in anames:
            ta = ta - kw['x']
        if 'w' not in anames:
            ta = ta / kw['w']

        # finally call the function
        mod = f(ta, **fkw)
        # recalcualte, making the peak maximize at x
        mod = f(ta + ta[mod.argmax()], **fkw)
        return kw['v'] + kw['h'] / max(mod) * mod

    args = set(['v', 'h', 'x', 'w'])
    anames, _, _, _ = inspect.getargspec(f)
    wrapped_f._peakargs = list(args.union([a for a in anames
                                           if a not in ('t', 'r')]))
    return wrapped_f