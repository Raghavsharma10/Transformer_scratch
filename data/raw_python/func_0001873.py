def load_json(path):
    # type: (str) -> OrderedDict
    """Load JSON file into an ordered dictionary

    Args:
        path (str): Path to JSON file

    Returns:
        OrderedDict: Ordered dictionary containing loaded JSON file
    """
    with open(path, 'rt') as f:
        jsondict = json.loads(f.read(), object_pairs_hook=OrderedDict)
    if not jsondict:
        raise (LoadError('JSON file: %s is empty!' % path))
    return jsondict