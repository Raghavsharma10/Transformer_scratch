def config_dict(config_file=None, auto_find=False, verify=True, **cfg_options):
    """
    Return configuration options as dictionary. Accepts either a single
    config file or a list of files. Auto find will search for all .cfg, .config
    and .ini in the execution directory and package root (unsafe but handy).

    .. code:: python

        reusables.config_dict(os.path.join("test", "data", "test_config.ini"))
        # {'General': {'example': 'A regular string'},
        #  'Section 2': {'anint': '234',
        #                'examplelist': '234,123,234,543',
        #                'floatly': '4.4',
        #                'my_bool': 'yes'}}


    :param config_file: path or paths to the files location
    :param auto_find: look for a config type file at this location or below
    :param verify: make sure the file exists before trying to read
    :param cfg_options: options to pass to the parser
    :return: dictionary of the config files
    """
    if not config_file:
        config_file = []

    cfg_parser = ConfigParser.ConfigParser(**cfg_options)
    cfg_files = []

    if config_file:
        if not isinstance(config_file, (list, tuple)):
            if isinstance(config_file, str):
                cfg_files.append(config_file)
            else:
                raise TypeError("config_files must be a list or a string")
        else:
            cfg_files.extend(config_file)

    if auto_find:
        cfg_files.extend(find_files_list(
            current_root if isinstance(auto_find, bool) else auto_find,
            ext=(".cfg", ".config", ".ini")))

    logger.info("config files to be used: {0}".format(cfg_files))

    if verify:
        cfg_parser.read([cfg for cfg in cfg_files if os.path.exists(cfg)])
    else:
        cfg_parser.read(cfg_files)

    return dict((section, dict(cfg_parser.items(section)))
                for section in cfg_parser.sections())