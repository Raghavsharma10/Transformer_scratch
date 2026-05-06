def setup_config(epab_version: str):
    """
    Set up elib_config package

    :param epab_version: installed version of EPAB as as string
    """
    logger = logging.getLogger('EPAB')
    logger.debug('setting up config')
    elib_config.ELIBConfig.setup(
        app_name='EPAB',
        app_version=epab_version,
        config_file_path='pyproject.toml',
        config_sep_str='__',
        root_path=['tool', 'epab']
    )
    elib_config.write_example_config('pyproject.toml.example')
    if not pathlib.Path('pyproject.toml').exists():
        raise FileNotFoundError('pyproject.toml')
    elib_config.validate_config()