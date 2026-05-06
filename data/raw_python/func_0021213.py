def beautify(filename=None, json_str=None):
    """Beautify JSON string or file.

    Keyword arguments:
    :param filename: use its contents as json string instead of
    json_str param.
    :param json_str: json string to be beautified.
    """
    if filename is not None:
        with open(filename) as json_file:
            json_str = json.load(json_file)

    return  json.dumps(json_str, indent=4, sort_keys=True)