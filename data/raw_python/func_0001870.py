def save_yaml(dictionary, path, pretty=False, sortkeys=False):
    # type: (Dict, str, bool, bool) -> None
    """Save dictionary to YAML file preserving order if it is an OrderedDict

    Args:
        dictionary (Dict): Python dictionary to save
        path (str): Path to YAML file
        pretty (bool): Whether to pretty print. Defaults to False.
        sortkeys (bool): Whether to sort dictionary keys. Defaults to False.

    Returns:
        None
    """
    if sortkeys:
        dictionary = dict(dictionary)
    with open(path, 'w') as f:
        if pretty:
            pyaml.dump(dictionary, f)
        else:
            yaml.dump(dictionary, f, default_flow_style=None, Dumper=yamlloader.ordereddict.CDumper)