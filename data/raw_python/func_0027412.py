def filter_params(params):
    """
    convert dict value if value is bool type,
    False -> "false"
    True -> "true"
    """
    if params is not None:
        new_params = copy.deepcopy(params)
        new_params = dict((k, v) for k, v in new_params.items() if v is not None)
        for key, value in new_params.items():
            if isinstance(value, bool):
                new_params[key] = "true" if value else "false"
        return new_params