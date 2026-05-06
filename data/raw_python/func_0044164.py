def init(config):
    """
    Initialise ~./sedge/config file if none exists.
    Good for first time sedge usage
    """
    from pkg_resources import resource_stream
    import shutil

    config_file = Path(config.config_file)
    if config_file.is_file():
        click.echo('{} already exists, maybe you want $ sedge update'.format(config_file))
        sys.exit()

    config_file.parent.mkdir(parents=True, exist_ok=True)
    with resource_stream(__name__, 'sedge_template.conf') as src_stream:
        with open(config.config_file, 'wb') as target_stream:
            shutil.copyfileobj(src_stream, target_stream)