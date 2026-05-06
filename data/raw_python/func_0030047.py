def load_config(path=None, load_user=True):
    """
    Load configuration information from a config directory. Tries directories in this order:

    - A path provided as an argument
    - A path specified by the AMBRY_CONFIG environmenal variable
    - ambry in a path specified by the VIRTUAL_ENV environmental variable
    - /etc/ambry
    - ~/ambry

    :param path: An iterable of additional paths to load.
    :return: An `AttrDict` of configuration information
    """

    from os.path import getmtime

    config = AttrDict()

    if not path:
        path = ROOT_DIR

    config_file = find_config_file(CONFIG_FILE, extra_path=path, load_user=load_user)

    if os.path.exists(config_file):

        config.update_yaml(config_file)
        config.loaded = [config_file, getmtime(config_file)]

    else:
        # Probably never get here, since the find_config_dir would have thrown a ConfigurationError
        config = AttrDict()
        config.loaded = [None, 0]

    return config