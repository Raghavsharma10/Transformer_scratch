def MAX(values, *others):
    """
    DECISIVE MAX
    :param values:
    :param others:
    :return:
    """

    if others:
        from mo_logs import Log
        Log.warning("Calling wrong")
        return MAX([values] + list(others))

    output = Null
    for v in values:
        if v == None:
            continue
        elif output == None or v > output:
            output = v
        else:
            pass
    return output