def from_pypirc(pypi_repository):
    """ Load configuration from .pypirc file, cached to only run once """
    ret = {}
    pypirc_locations = PYPIRC_LOCATIONS
    for pypirc_path in pypirc_locations:
        pypirc_path = os.path.expanduser(pypirc_path)
        if os.path.isfile(pypirc_path):
            parser = configparser.SafeConfigParser()
            parser.read(pypirc_path)
            if 'distutils' not in parser.sections():
                continue
            if 'index-servers' not in parser.options('distutils'):
                continue
            if pypi_repository not in parser.get('distutils', 'index-servers'):
                continue
            if pypi_repository in parser.sections():
                for option in parser.options(pypi_repository):
                    ret[option] = parser.get(pypi_repository, option)
    if not ret:
        raise ConfigError(
            'repository does not appear to be configured in pypirc ({})'.format(pypi_repository) +
            ', remember that it needs an entry in [distutils] and its own section'
        )
    return ret