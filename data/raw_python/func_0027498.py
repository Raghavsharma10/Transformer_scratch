def pop_kwargs(kwargs,
               names_with_defaults,  # type: List[Tuple[str, Any]]
               allow_others=False
               ):
    """
    Internal utility method to extract optional arguments from kwargs.

    :param kwargs:
    :param names_with_defaults:
    :param allow_others: if False (default) then an error will be raised if kwargs still contains something at the end.
    :return:
    """
    all_arguments = []
    for name, default_ in names_with_defaults:
        try:
            val = kwargs.pop(name)
        except KeyError:
            val = default_
        all_arguments.append(val)

    if not allow_others and len(kwargs) > 0:
        raise ValueError("Unsupported arguments: %s" % kwargs)

    if len(names_with_defaults) == 1:
        return all_arguments[0]
    else:
        return all_arguments