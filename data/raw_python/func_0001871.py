def save_json(dictionary, path, pretty=False, sortkeys=False):
    # type: (Dict, str, bool, bool) -> None
    """Save dictionary to JSON file preserving order if it is an OrderedDict

    Args:
        dictionary (Dict): Python dictionary to save
        path (str): Path to JSON file
        pretty (bool): Whether to pretty print. Defaults to False.
        sortkeys (bool): Whether to sort dictionary keys. Defaults to False.

    Returns:
        None
    """
    with open(path, 'w') as f:
        if pretty:
            indent = 2
            separators = (',', ': ')
        else:
            indent = None
            separators = (', ', ': ')
        json.dump(dictionary, f, indent=indent, sort_keys=sortkeys, separators=separators)