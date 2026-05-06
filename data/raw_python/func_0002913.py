def save_list(key, *values):
    """Convert the given list of parameters to a JSON object.

    JSON object is of the form:
    { key: [values[0], values[1], ... ] },
    where values represent the given list of parameters.

    """
    return json.dumps({key: [_get_json(value) for value in values]})