def install_python_module_locally(name):
    """ instals a python module using pip """
    with settings(hide('everything'),
                  warn_only=False, capture=True):
        # convert 0 into True, any errors will always raise an exception
        print(not bool(local('pip --quiet install %s' % name).return_code))
        return not bool(
            local('pip --quiet install %s' % name).return_code)