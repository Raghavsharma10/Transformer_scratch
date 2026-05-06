def format_exception(etype, value, tback, limit=None):
    """
    Python 2 compatible version of traceback.format_exception
    Accepts negative limits like the Python 3 version
    """

    rtn = ['Traceback (most recent call last):\n']

    if limit is None or limit >= 0:
        rtn.extend(traceback.format_tb(tback, limit))
    else:
        rtn.extend(traceback.format_list(traceback.extract_tb(tback)[limit:]))

    rtn.extend(traceback.format_exception_only(etype, value))

    return rtn