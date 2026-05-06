def load_remotes(extra_path=None, load_user=True):
    """Load the YAML remotes file, which sort of combines the Accounts file with part of the
    remotes sections from the main config

    :return: An `AttrDict`
    """

    from os.path import getmtime

    try:
        remotes_file = find_config_file(REMOTES_FILE, extra_path=extra_path, load_user=load_user)
    except ConfigurationError:
        remotes_file = None


    if remotes_file is not None and os.path.exists(remotes_file):
        config = AttrDict()
        config.update_yaml(remotes_file)

        if not 'remotes' in config:
            config.remotes = AttrDict()

        config.remotes.loaded = [remotes_file, getmtime(remotes_file)]

        return config
    else:
        return None