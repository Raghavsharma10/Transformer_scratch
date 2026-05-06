def _prep_smooth(t, y, dy, span, t_out, span_out, period):
    """Private function to prepare & check variables for smooth utilities"""

    # If period is provided, sort by phases. Otherwise sort by t
    if period:
        t = t % period
        if t_out is not None:
            t_out = t_out % period

    t, y, dy = validate_inputs(t, y, dy, sort_by=t)

    if span_out is not None:
        if t_out is None:
            raise ValueError("Must specify t_out when span_out is given")
        if span is not None:
            raise ValueError("Must specify only one of span, span_out")
        span, t_out = np.broadcast_arrays(span_out, t_out)
        indices = np.searchsorted(t, t_out)
    elif span is None:
        raise ValueError("Must specify either span_out or span")
    else:
        indices = None

    return t, y, dy, span, t_out, span_out, indices