def load_path(path, overrides=None, **kwargs):
    """
    Convenience function for loading a YAML configuration from a file.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    overrides : dict, optional
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified an
        Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    f = open(path, 'r')
    content = ''.join(f.readlines())
    f.close()

    if not isinstance(content, str):
        raise AssertionError("Expected content to be of type str but it is "+str(type(content)))

    return load(content, **kwargs)