def get_dimension(data):
    """
    Get dimension of the data passed by argument independently if it's an
    arrays or dictionaries
    """
    result = [0, 0]

    if isinstance(data, list):
        result = get_dimension_array(data)

    elif isinstance(data, dict):
        result = get_dimension_dict(data)

    return result