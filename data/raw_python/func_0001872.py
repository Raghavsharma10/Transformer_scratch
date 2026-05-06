def load_yaml(path):
    # type: (str) -> OrderedDict
    """Load YAML file into an ordered dictionary

    Args:
        path (str): Path to YAML file

    Returns:
        OrderedDict: Ordered dictionary containing loaded YAML file
    """
    with open(path, 'rt') as f:
        yamldict = yaml.load(f.read(), Loader=yamlloader.ordereddict.CSafeLoader)
    if not yamldict:
        raise (LoadError('YAML file: %s is empty!' % path))
    return yamldict