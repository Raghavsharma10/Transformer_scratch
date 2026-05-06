def parse_json(content):
    """Tries to parse a string into a json object.

    This also performs a trim of all values, recursively removing leading and trailing whitespace.

    Parameters
    ----------
    content: A JSON format string.

    Returns
    -------
    obj:
        The object represented by the json string.

    Raises
    ------
    InvalidContent
        If the content is not a valid json string.
    """
    try:
        json_content = json.loads(content)
        return _recursive_strip(json_content)
    except json.JSONDecodeError:
        raise InvalidContent("content is not a json string.")