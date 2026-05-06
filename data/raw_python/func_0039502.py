def cli(dirty, stash):
    """
    This is a tool that handles all the tasks to build a Python application

    This tool is installed as a setuptools entry point, which means it should be accessible from your terminal once
    this application is installed in develop mode.
    """
    _setup_logging()

    LOGGER.info('EPAB %s', __version__)
    LOGGER.info('Running in %s', os.getcwd())

    CTX.repo = epab.utils.Repo()
    CTX.repo.ensure()
    CTX.stash = stash
    for filename in _GIT_IGNORE:
        epab.utils.add_to_gitignore(filename)
    if not dirty and CTX.repo.is_dirty():
        LOGGER.error('Repository is dirty')
        sys.exit(-1)