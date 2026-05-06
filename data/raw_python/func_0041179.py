def exclude_values(values, args):
    """
    Exclude data with specific value.
    =============   =============   =======================================
    Parameter       Type            Description
    =============   =============   =======================================
    values          list            values where exclude elements
    args            list or dict    elements to exclude
    =============   =============   =======================================
    Returns: vakues without excluded elements
    """

    if isinstance(args, dict):
        return {
            key: value
            for key, value in (
                (k, exclude_values(values, v)) for (k, v) in args.items())
            if value not in values
        }
    elif isinstance(args, list):
        return [
            item
            for item in [exclude_values(values, i) for i in args]
            if item not in values
        ]

    return args