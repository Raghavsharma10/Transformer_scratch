def arch():
    """ returns the current cpu archictecture """
    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):
        result = sudo('rpm -E %dist').strip()
    return result