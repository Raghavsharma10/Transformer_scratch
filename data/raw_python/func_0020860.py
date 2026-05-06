def extract_params(params):
    """
    Extracts the values of a set of parameters, recursing into nested dictionaries.
    """
    values = []
    if isinstance(params, dict):
        for key, value in params.items():
            values.extend(extract_params(value))
    elif isinstance(params, list):
        for value in params:
            values.extend(extract_params(value))
    else:
        values.append(params)
    return values