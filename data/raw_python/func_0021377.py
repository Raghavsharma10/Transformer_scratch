def linear_smooth(t, y, dy, span=None, cv=True,
                  t_out=None, span_out=None, period=None):
    """Perform a linear smooth of the data

    Parameters
    ----------
    t, y, dy : array_like
        time, value, and error in value of the input data
    span : array_like
        the integer spans of the data
    cv : boolean (default=True)
        if True, treat the problem as a cross-validation, i.e. don't use
        each point in the evaluation of its own smoothing.
    t_out : array_like (optional)
        the output times for the moving averages
    span_out : array_like (optional)
        the spans associated with the output times t_out
    period : float
        if provided, then consider the inputs periodic with the given period

    Returns
    -------
    y_smooth : array_like
        smoothed y values at each time t or t_out
    """
    t_input = t
    prep = _prep_smooth(t, y, dy, span, t_out, span_out, period)
    t, y, dy, span, t_out, span_out, indices = prep
    if period:
        t_input = np.asarray(t_input) % period

    w = 1. / (dy ** 2)
    w, yw, tw, tyw, ttw = windowed_sum([w, y * w, w, y * w, w], t=t,
                                       tpowers=[0, 0, 1, 1, 2],
                                       span=span, indices=indices,
                                       subtract_mid=cv, period=period)

    denominator = (w * ttw - tw * tw)
    slope = (tyw * w - tw * yw)
    intercept = (ttw * yw - tyw * tw)

    if np.any(denominator == 0):
        raise ValueError("Zero denominator in linear smooth. This usually "
                         "indicates that the input contains duplicate points.")

    if t_out is None:
        return (slope * t_input + intercept) / denominator
    elif span_out is not None:
        return (slope * t_out + intercept) / denominator
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return (slope[i] * t_out + intercept[i]) / denominator[i]