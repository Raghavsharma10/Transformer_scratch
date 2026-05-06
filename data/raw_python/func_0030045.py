def load_accounts(extra_path=None, load_user=True):
    """Load the yaml account files

    :param load_user:
    :return: An `AttrDict`
    """

    from os.path import getmtime


    try:
        accts_file = find_config_file(ACCOUNTS_FILE, extra_path=extra_path, load_user=load_user)
    except ConfigurationError:
        accts_file = None

    if accts_file is not None and os.path.exists(accts_file):
        config = AttrDict()
        config.update_yaml(accts_file)

        if not 'accounts' in config:
            config.remotes = AttrDict()

        config.accounts.loaded = [accts_file, getmtime(accts_file)]
        return config
    else:
        return None