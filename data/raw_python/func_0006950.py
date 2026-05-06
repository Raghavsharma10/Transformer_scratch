def os_release(username, ip_address):
    """ returns /etc/os-release in a dictionary """
    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):

        _os_release = {}
        with settings(host_string=username + '@' + ip_address):
            data = run('cat /etc/os-release')
        for line in data.split('\n'):
            if not line:
                continue
            parts = line.split('=')
            if len(parts) == 2:
                _os_release[parts[0]] = parts[1].strip('\n\r"')

        return _os_release