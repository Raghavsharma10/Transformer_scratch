def load(stream, overrides=None, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object.

    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object
        supporting the .read() interface.
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

    global is_initialized
    if not is_initialized:
        initialize()

    if isinstance(stream, basestring):
        string = stream
    else:
        string = '\n'.join(stream.readlines())

    # processed_string = preprocess(string)

    proxy_graph = yaml.load(string, **kwargs)

    from . import init
    init_dict = proxy_graph.get('init', {})
    init(**init_dict)
    
    if overrides is not None:
        handle_overrides(proxy_graph, overrides)
    return instantiate_all(proxy_graph)