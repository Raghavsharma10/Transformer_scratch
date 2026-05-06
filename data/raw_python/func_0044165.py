def update(config):
    """
    Update ssh config from sedge specification
    """

    def write_to(out):
        engine.output(out)

    config_file = Path(config.config_file)
    if not config_file.is_file():
        click.echo('No file {} '.format(config_file), err=True)
        sys.exit()

    library = KeyLibrary(config.key_directory)
    with config_file.open() as fd:
        engine = SedgeEngine(library, fd, not config.no_verify, url=config.config_file)

    if config.output_file == '-':
        write_to(ConfigOutput(sys.stdout))
        return

    if not check_or_confirm_overwrite(config.output_file):
        click.echo('Aborting.', err=True)
        sys.exit(1)

    tmp_file = NamedTemporaryFile(mode='w', dir=os.path.dirname(config.output_file), delete=False)
    try:
        tmp_file.file.write(sedge_config_header.format(config.config_file))
        write_to(ConfigOutput(tmp_file.file))
        tmp_file.close()
        if config.verbose:
            diff_config_changes(config.output_file, tmp_file.name)
        os.rename(tmp_file.name, config.output_file)
    except:
        os.unlink(tmp_file.name)
        raise