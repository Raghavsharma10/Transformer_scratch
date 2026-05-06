def linux_distribution():
    """ returns the linux distribution in lower case """
    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):
        data = os_release()
        return(data['ID'])