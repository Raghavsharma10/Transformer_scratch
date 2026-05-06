def gen_pypirc(username=None, password=None):
    # type: (str, str) -> None
    """ Generate ~/.pypirc with the given credentials.

    Useful for CI builds. Can also get credentials through env variables
    ``PYPI_USER`` and ``PYPI_PASS``.

    Args:
        username (str):
            pypi username. If not given it will try to take it from the
            `` PYPI_USER`` env variable.
        password (str):
            pypi password. If not given it will try to take it from the
            `` PYPI_PASS`` env variable.
    """
    path = join(conf.getenv('HOME'), '.pypirc')
    username = username or conf.getenv('PYPI_USER', None)
    password = password or conf.getenv('PYPI_PASS', None)

    if username is None or password is None:
        log.err("You must provide $PYPI_USER and $PYPI_PASS")
        sys.exit(1)

    log.info("Generating <94>{}".format(path))

    fs.write_file(path, util.remove_indent('''
        [distutils]
        index-servers = pypi
        
        [pypi]
        repository: https://upload.pypi.org/legacy/
        username: {username}
        password: {password}
        
    '''.format(
        username=username,
        password=password
    )))