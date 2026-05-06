def load(path=None, root=None, db=None, load_user=True):
    "Load all of the config files. "

    config = load_config(path, load_user=load_user)

    remotes = load_remotes(path, load_user=load_user)

    # The external file overwrites the main config
    if remotes:
        if not 'remotes' in config:
            config.remotes = AttrDict()

        for k, v in remotes.remotes.items():
            config.remotes[k] = v

    accounts = load_accounts(path, load_user=load_user)

    # The external file overwrites the main config
    if accounts:
        if not 'accounts' in config:
            config.accounts = AttrDict()
        for k, v in accounts.accounts.items():
            config.accounts[k] = v

    update_config(config)

    if root:
        config.library.filesystem_root = root

    if db:
        config.library.database = db

    return config