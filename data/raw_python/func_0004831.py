def _try_pydatetime(x):
    """Try to convert to pandas objects to datetimes.

    Plotly doesn't know how to handle them.
    """
    try:
        # for datetimeindex
        x = [y.isoformat() for y in x.to_pydatetime()]
    except AttributeError:
        pass
    try:
        # for generic series
        x = [y.isoformat() for y in x.dt.to_pydatetime()]
    except AttributeError:
        pass
    return x