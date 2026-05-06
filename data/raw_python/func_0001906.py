def load_yaml_file(file):
    """
    Load data from yaml file

    :param file: Readable object or path to file
    :type file: FileIO | str | unicode
    :return: Yaml data
    :rtype: None | int | float | str | unicode | list | dict
    """
    if not hasattr(file, "read"):
        with io.open(file, "r", encoding="utf-8") as f:
            return yaml.load(f, yaml.FullLoader)
    return yaml.load(file, yaml.FullLoader)