def install_python_module(name):
    """ instals a python module using pip """

    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=False, capture=True):
        run('pip --quiet install %s' % name)