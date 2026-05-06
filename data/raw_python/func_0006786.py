def os_release():
    """ returns /etc/os-release in a dictionary """
    with settings(hide('warnings', 'running', 'stderr'),
                  warn_only=True, capture=True):

        release = {}
        data = run('cat /etc/os-release')
        for line in data.split('\n'):
            if not line:
                continue
            parts = line.split('=')
            if len(parts) == 2:
                release[parts[0]] = parts[1].strip('\n\r"')

        return release