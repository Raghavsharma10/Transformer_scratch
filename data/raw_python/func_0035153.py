def create_virtualenv(pkg, repo_dest, python):
    """Creates a virtualenv within which to install your new application."""
    workon_home = os.environ.get('WORKON_HOME')
    venv_cmd = find_executable('virtualenv')
    python_bin = find_executable(python)
    if not python_bin:
        raise EnvironmentError('%s is not installed or not '
                               'available on your $PATH' % python)
    if workon_home:
        # Can't use mkvirtualenv directly here because relies too much on
        # shell tricks. Simulate it:
        venv = os.path.join(workon_home, pkg)
    else:
        venv = os.path.join(repo_dest, '.virtualenv')
    if venv_cmd:
        if not verbose:
            log.info('Creating virtual environment in %r' % venv)
        args = ['--python', python_bin, venv]
        if not verbose:
            args.insert(0, '-q')
        subprocess.check_call([venv_cmd] + args)
    else:
        raise EnvironmentError('Could not locate the virtualenv. Install with '
                               'pip install virtualenv.')
    return venv