def config_default(dest):
    """ Create a default configuration file.

    \b
    DEST: Path or file name for the configuration file.
    """
    conf_path = Path(dest).resolve()
    if conf_path.is_dir():
        conf_path = conf_path / LIGHTFLOW_CONFIG_NAME

    conf_path.write_text(Config.default())
    click.echo('Configuration written to {}'.format(conf_path))