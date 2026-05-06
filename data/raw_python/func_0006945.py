def install_python_module_locally(name):
    """ instals a python module using pip """
    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=False, capture=True):
        local('pip --quiet install %s' % name)