def options(f):
    """
    Shared options, used by all bartender commands
    """

    f = click.option('--config', envvar='VODKA_HOME', default=click.get_app_dir('vodka'), help="location of config file")(f)
    return f