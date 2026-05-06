def lsb_release():
    """ returns /etc/lsb-release in a dictionary """
    with settings(hide('warnings', 'running'), capture=True):

        _lsb_release = {}
        data = sudo('cat /etc/lsb-release')
        for line in data.split('\n'):
            if not line:
                continue
            parts = line.split('=')
            if len(parts) == 2:
                _lsb_release[parts[0]] = parts[1].strip('\n\r"')

        return _lsb_release