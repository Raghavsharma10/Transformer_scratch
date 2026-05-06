def update_config(config, use_environ=True):
    """Update the configuration from environmental variables. Updates:

    - config.library.database from the AMBRY_DB environmental variable.
    - config.library.filesystem_root from the AMBRY_ROOT environmental variable.
    - config.accounts.password from the AMBRY_PASSWORD  environmental variable.

    :param config: An `attrDict` of configuration information.
    """
    from ambry.util import select_from_url


    try:
        _ = config.library
    except KeyError:
        config.library = AttrDict()

    try:
        _ = config.filesystem
    except KeyError:
        config.filesystem = AttrDict()

    try:
        _ = config.accounts
    except KeyError:
        config.accounts = AttrDict()

    if not config.accounts.get('loaded'):
        config.accounts.loaded = [None, 0]

    try:
        _ = config.accounts.password
    except KeyError:
        config.accounts.password = None

    try:
        _ = config.remotes
    except KeyError:
        config.remotes = AttrDict()  # Default empty

    if not config.remotes.get('loaded'):
        config.remotes.loaded = [None, 0]

    if use_environ:
        if os.getenv(ENVAR.DB):
            config.library.database = os.getenv(ENVAR.DB)

        if os.getenv(ENVAR.ROOT):
            config.library.filesystem_root = os.getenv(ENVAR.ROOT)

        if os.getenv(ENVAR.PASSWORD):
            config.accounts.password = os.getenv(ENVAR.PASSWORD)

    # Move any remotes that were configured under the library to the remotes section

    try:
        for k, v in config.library.remotes.items():
            config.remotes[k] = {
                'url': v
            }

        del config.library['remotes']

    except KeyError as e:
        pass

    # Then move any of the account entries that are linked to remotes into the remotes.

    try:
        for k, v in config.remotes.items():
            if 'url' in v:
                host = select_from_url(v['url'], 'netloc')
                if host in config.accounts:
                    config.remotes[k].update(config.accounts[host])
                    del config.accounts[host]

    except KeyError:
        pass


    # Set a default for the library database
    try:
        _ = config.library.database
    except KeyError:
        config.library.database = 'sqlite:///{root}/library.db'

    # Raise exceptions on missing items
    checks = [
        'config.library.filesystem_root',
    ]

    for check in checks:
        try:
            _ = eval(check)
        except KeyError:
            raise ConfigurationError("Configuration is missing '{}'; loaded from {} "
                                     .format(check, config.loaded[0]))

    _, config.library.database = normalize_dsn_or_dict(config.library.database)

    for k, v in filesystem_defaults.items():
        if k not in config.filesystem:
            config.filesystem[k] = v

    config.modtime = max(config.loaded[1], config.remotes.loaded[1], config.accounts.loaded[1])