def _homogenize_data_filter(dfilter):
    """
    Make data filter definition consistent.

    Create a tuple where first element is the row filter and the second element
    is the column filter
    """
    if isinstance(dfilter, tuple) and (len(dfilter) == 1):
        dfilter = (dfilter[0], None)
    if (dfilter is None) or (dfilter == (None, None)) or (dfilter == (None,)):
        dfilter = (None, None)
    elif isinstance(dfilter, dict):
        dfilter = (dfilter, None)
    elif isinstance(dfilter, (list, str)) or (
        isinstance(dfilter, int) and (not isinstance(dfilter, bool))
    ):
        dfilter = (None, dfilter if isinstance(dfilter, list) else [dfilter])
    elif isinstance(dfilter[0], dict) or (
        (dfilter[0] is None) and (not isinstance(dfilter[1], dict))
    ):
        pass
    else:
        dfilter = (dfilter[1], dfilter[0])
    return dfilter