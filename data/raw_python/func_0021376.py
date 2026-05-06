def moving_average_smooth(t, y, dy, span=None, cv=True,
                          t_out=None, span_out=None, period=None):
    """Perform a moving-average smooth of the data

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
        smoothed y values at each time t (or t_out)
    """
    prep = _prep_smooth(t, y, dy, span, t_out, span_out, period)
    t, y, dy, span, t_out, span_out, indices = prep

    w = 1. / (dy ** 2)
    w, yw = windowed_sum([w, y * w], t=t, span=span, subtract_mid=cv,
                         indices=indices, period=period)

    if t_out is None or span_out is not None:
        return yw / w
    else:
        i = np.minimum(len(t) - 1, np.searchsorted(t, t_out))
        return yw[i] / w[i]