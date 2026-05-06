def get_lint_config(config_path=None):
    """ Tries loading the config from the given path. If no path is specified, the default config path
    is tried, and if that is not specified, we the default config is returned. """
    # config path specified
    if config_path:
        config = LintConfig.load_from_file(config_path)
        click.echo("Using config from {0}".format(config_path))
    # default config path
    elif os.path.exists(DEFAULT_CONFIG_FILE):
        config = LintConfig.load_from_file(DEFAULT_CONFIG_FILE)
        click.echo("Using config from {0}".format(DEFAULT_CONFIG_FILE))
    # no config file
    else:
        config = LintConfig()

    return config