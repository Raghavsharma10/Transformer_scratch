def response_list(data, key):
    """Obtain the relevant response data in a list.

    If the response does not already contain the result in a list, a new one
    will be created to ease iteration in the parser methods.

    Args:
        data (dict): API response.
        key (str): Attribute of the response that contains the result values.

    Returns:
        List of response items (usually dict) or None if the key is not present.
    """
    if key not in data:
        return None

    if isinstance(data[key], list):
        return data[key]

    else:
        return [data[key],]