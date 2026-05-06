def config_namespace(config_file=None, auto_find=False,
                     verify=True, **cfg_options):
    """
    Return configuration options as a Namespace.

    .. code:: python

        reusables.config_namespace(os.path.join("test", "data",
                                                "test_config.ini"))
        # <Namespace: {'General': {'example': 'A regul...>


    :param config_file: path or paths to the files location
    :param auto_find: look for a config type file at this location or below
    :param verify: make sure the file exists before trying to read
    :param cfg_options: options to pass to the parser
    :return: Namespace of the config files
    """
    return ConfigNamespace(**config_dict(config_file, auto_find,
                                         verify, **cfg_options))