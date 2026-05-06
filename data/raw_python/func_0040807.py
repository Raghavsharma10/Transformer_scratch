def kwargs_helper(kwargs):
    """This function preprocesses the kwargs dictionary to sanitize it."""

    args = []
    for param, value in kwargs.items():
        param = kw_subst.get(param, param)
        args.append((param, value))
    return args